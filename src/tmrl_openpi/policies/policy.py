from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from tmrl_openpi import transforms as _transforms
from tmrl_openpi.models import model as _model
from tmrl_openpi.shared import array_typing as at
from tmrl_openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._device = pytorch_device

        if is_pytorch:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._get_prefix_rep = model.get_prefix_rep
            self._ah = model.config.action_horizon
            self._ad = model.config.action_dim
        else:
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._invert_actions = nnx_utils.module_jit(model.invert_actions)
            self._rng = rng or jax.random.key(0)
            self._ah = model.action_horizon
            self._ad = model.action_dim

    def _to_np(self, x) -> np.ndarray:
        """Any array-like -> numpy."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _to_backend(self, arr: np.ndarray):
        """numpy -> backend tensor."""
        if self._is_pytorch_model:
            return torch.from_numpy(arr).to(self._device)
        return jnp.asarray(arr)

    def _prepare_inputs(self, obs: dict) -> tuple[dict, bool, int]:
        """Transform obs, convert to backend tensors, add batch dim if needed."""
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        batched = inputs["state"].ndim > 1
        bs = inputs["state"].shape[0] if batched else 1

        def _convert(x):
            if x is None:
                return None
            if self._is_pytorch_model:
                t = (
                    x.to(self._device)
                    if isinstance(x, torch.Tensor)
                    else torch.from_numpy(np.asarray(x)).to(self._device)
                )
                return t if batched else t.unsqueeze(0)
            a = jnp.asarray(x)
            return a if batched else a[np.newaxis]

        inputs = jax.tree.map(_convert, inputs)

        # Permute images for PyTorch (NHWC -> NCHW)
        if self._is_pytorch_model:
            for cam in inputs["image"]:
                img = inputs["image"][cam]
                # Check if it's NHWC (where the last dim is 3 channels)
                if img.ndim == 4 and img.shape[-1] == 3:
                    inputs["image"][cam] = img.permute(0, 3, 1, 2)

        if batched:
            # Handle image masks
            for cam in inputs["image"]:
                m = inputs["image_mask"][cam]
                if m.ndim == 0:
                    inputs["image_mask"][cam] = m.expand(bs) if self._is_pytorch_model else jnp.broadcast_to(m, (bs,))

            # Ensure prompt fields match the batch dimension bs
            for k in ["tokenized_prompt", "tokenized_prompt_mask"]:
                if k in inputs and inputs[k] is not None and inputs[k].ndim == 1:
                    inputs[k] = (
                        inputs[k].unsqueeze(0).expand(bs, -1)
                        if self._is_pytorch_model
                        else jnp.broadcast_to(inputs[k], (bs, *inputs[k].shape))
                    )

        return inputs, batched, bs

    def _normalize_noise(self, noise, bs: int) -> np.ndarray:
        """Coerce to float32 numpy (bs, action_horizon, action_dim)."""
        arr = self._to_np(noise)
        if arr.ndim == 2:
            arr = np.broadcast_to(arr, (bs, self._ad)).copy()
            arr = np.repeat(arr[:, None, :], self._ah, axis=1)
        if arr.ndim != 3 or arr.shape[0] != bs:
            raise ValueError(f"Bad noise shape {arr.shape}, expected ({bs}, {self._ah}, {self._ad})")
        return arr.astype(np.float32, copy=False)

    def _normalize_time(self, tcont_context, bs: int, batched: bool) -> np.ndarray:
        """Coerce to float32 numpy (bs,) or (1,)."""
        arr = self._to_np(tcont_context).ravel()
        if batched:
            arr = np.broadcast_to(arr, (bs,)).copy()
        else:
            arr = arr[:1]
        return arr.astype(np.float32, copy=False)

    @override
    def infer(
        self,
        obs: dict,
        *,
        action_noise: np.ndarray | None = None,
        tcont_context: np.ndarray | None = None,
        context_noise: np.ndarray | None = None,
    ) -> dict:  # type: ignore[misc]
        inputs, batched, bs = self._prepare_inputs(obs)
        kw = dict(self._sample_kwargs)

        if action_noise is not None:
            kw["noise"] = self._to_backend(self._normalize_noise(action_noise, bs))
        if tcont_context is not None:
            kw["time_prefix"] = self._to_backend(self._normalize_time(tcont_context, bs, batched))
        if context_noise is not None:
            arr = np.repeat(self._to_np(context_noise)[:, None, :], 816, axis=1)  # TODO: hardcoded prefix seq len
            kw["noise_prefix"] = self._to_backend(arr)

        observation = _model.Observation.from_dict(inputs)
        t0 = time.monotonic()

        if self._is_pytorch_model:
            actions = self._sample_actions(self._device, observation, **kw)
        else:
            self._rng, rng = jax.random.split(self._rng)
            kw["rng"] = rng
            actions = self._sample_actions(observation, **kw)

        raw = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(
            lambda x: (
                np.asarray(x.detach().cpu()) if hasattr(x, "detach") else (np.asarray(x) if x is not None else None)
            ),
            raw,
        )
        if not batched:
            outputs = jax.tree.map(lambda x: x[0] if hasattr(x, "ndim") and x.ndim > 0 else x, outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}

        # Optional debug-only roundtrip check.
        # Keep this off by default because inversion is expensive and requires model-space
        # action shapes (e.g. [B, H, 32]), not post-transform env actions (e.g. [H, 8] for droid).
        # if bool(self._sample_kwargs.get("debug_invert_roundtrip", False)):
        # model_actions = raw["actions"]
        # logging.info("Starting invert_actions (debug roundtrip)")

        # if self._is_pytorch_model:
        #     target_noise = self._invert_actions(self._device, observation, model_actions)
        # else:
        #     self._rng, rng_inv = jax.random.split(self._rng)
        #     target_noise = self._invert_actions(rng_inv, observation, model_actions)

        # target_noise_np = np.asarray(target_noise)
        # print("target_noise norm:", np.linalg.norm(target_noise_np))
        # outputs["target_noise"] = target_noise_np

        # if self._is_pytorch_model:
        #     reconstructed_actions = self._sample_actions(self._device, observation, noise=target_noise)
        # else:
        #     self._rng, rng_sample = jax.random.split(self._rng)
        #     reconstructed_actions = self._sample_actions(observation, rng=rng_sample, noise=target_noise)

        # reconstructed_actions_np = np.asarray(reconstructed_actions)
        # print("reconstructed_actions norm:", np.linalg.norm(reconstructed_actions_np))
        # action_roundtrip_l1 = np.abs(np.asarray(model_actions) - reconstructed_actions_np).mean()
        # print("action round trip L1 error:", action_roundtrip_l1)

        # if self._is_pytorch_model:
        #     noise_from_reconstructed_actions = self._invert_actions(
        #         self._device, observation, reconstructed_actions
        #     )
        # else:
        #     self._rng, rng_inv2 = jax.random.split(self._rng)
        #     noise_from_reconstructed_actions = self._invert_actions(
        #         rng_inv2, observation, reconstructed_actions
        #     )

        # noise_from_reconstructed_actions_np = np.asarray(noise_from_reconstructed_actions)
        # print("noise_from_reconstructed_actions norm:", np.linalg.norm(noise_from_reconstructed_actions_np))
        # print("noise round trip L1 error:", np.abs(target_noise_np - noise_from_reconstructed_actions_np).mean())

        # outputs["reconstructed_actions"] = reconstructed_actions_np
        # outputs["noise_from_reconstructed_actions"] = noise_from_reconstructed_actions_np

        return outputs

    @override
    def get_prefix_rep(self, obs: dict):
        inputs, _, _ = self._prepare_inputs(obs)
        return self._get_prefix_rep(_model.Observation.from_dict(inputs))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy
        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1
        np.save(output_path, np.asarray(data))
        return results
