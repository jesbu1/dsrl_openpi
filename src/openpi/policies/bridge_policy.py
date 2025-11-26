import dataclasses
import einops
import numpy as np

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Any
from typing_extensions import override

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.transforms as transforms

# Assumed action dimension for USC WidowX (6 joints + 1 gripper)


def _decode_bridge(data: dict) -> dict:
    state = np.asarray(data["state"])
    if "actions" in data:
        actions = np.asarray(data["actions"])
    else:
        actions = None

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            if np.max(img) <= 1:
                img = (255 * img).astype(np.uint8)
        if img.shape[0] == 3:
            img = einops.rearrange(img, "c h w -> h w c")
        return img

    if "camera_present" in data:
        camera_present = np.asarray(data["camera_present"])
    else:
        camera_present = [1]
    data["camera_present"] = camera_present

    for k, v in data.items():
        if "image" in k:
            data[k] = convert_image(v)
    data["state"] = state
    if actions is not None:
        data["actions"] = actions
    return data


@dataclasses.dataclass
class BridgeInputs(transforms.DataTransformFn):
    """Prepares BRIDGE inputs for the model.

    Assumes input keys: 'observation.images.image_0',..., 'state', 'actions'.
    The 'state' is expected to be a 7-dim vector (6 joint angles + 1 gripper state).
    The 'action' is expected to be a 7-dim vector (6 joint actions + 1 gripper action).
    """

    # Determines which model will be used.
    model_type: _model.ModelType
    use_delta_actions: bool = False  # already in delta space
    how_many_cameras: int = 1

    @override
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0 or self.model_type == _model.ModelType.PI05

        # Ensure expected keys are present
        required_keys = {
            "observation.images.image_0",
            "state",
        }
        if not required_keys.issubset(sample.keys()):
            raise ValueError(f"Missing required keys. Found: {sample.keys()}, Required: {required_keys}")

        # Create inputs dict. Do not change the keys in the dict below.
        # convert images to numpy arrays
        sample = _decode_bridge(sample)

        # other keys include: "observation.images.image_1", "observation.images.image_2", "observation.images.image_3". Check camera_present to see which ones are present and sample.
        if "camera_present" in sample:
            # old bridge sampling
            available_cameras = np.count_nonzero(sample["camera_present"])
            assert available_cameras > 0, "No cameras present in the sample"
            camera_idx_to_include = np.arange(available_cameras)[: self.how_many_cameras]
        else:
            # new just 1 cam bridge dataset
            camera_idx_to_include = [0]
            self.how_many_cameras = 1

        zero_image = np.zeros_like(sample["observation.images.image_0"])
        inputs = {
            "state": sample["state"],
            "image": {
                "base_0_rgb": zero_image,
                "base_1_rgb": zero_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "left_wrist_0_rgb": zero_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": zero_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "base_1_rgb": np.False_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }
        # now include the cameras in the required_keys
        camera_images = [f"observation.images.image_{i}" for i in camera_idx_to_include]

        if self.how_many_cameras == 2:
            inputs["image"]["base_0_rgb"] = sample[camera_images[0]]
            inputs["image_mask"]["base_0_rgb"] = np.True_
            if len(camera_images) > 1:
                inputs["image"]["base_1_rgb"] = sample[camera_images[1]]
                inputs["image_mask"]["base_1_rgb"] = np.True_
            else:
                inputs["image"]["base_1_rgb"] = inputs["image"]["base_0_rgb"]
                inputs["image_mask"]["base_1_rgb"] = inputs["image_mask"]["base_0_rgb"]
        elif self.how_many_cameras == 1:
            inputs["image"]["base_0_rgb"] = sample[camera_images[0]]
            inputs["image_mask"]["base_0_rgb"] = np.True_

        if self.model_type == _model.ModelType.PI0:
            # no base_1_rgb for pi0
            inputs["image"]["right_wrist_0_rgb"] = inputs["image"]["base_1_rgb"]
            inputs["image_mask"]["right_wrist_0_rgb"] = inputs["image_mask"]["base_1_rgb"]
            del inputs["image"]["base_1_rgb"]
            del inputs["image_mask"]["base_1_rgb"]
        if "actions" in sample:
            inputs["actions"] = sample["actions"]
        if "prompt" in sample:
            inputs["prompt"] = sample["prompt"]

        return inputs


@dataclasses.dataclass
class BridgeOutputs(transforms.DataTransformFn):
    """Converts model outputs back to BRIDGE action space."""

    use_delta_actions: bool = False

    @override
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {"actions": np.asarray(sample["actions"][:, :7])}