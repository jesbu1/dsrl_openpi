import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from tmrl_openpi.models import model as _model
import tmrl_openpi.models.siglip as _siglip
from tmrl_openpi.shared import array_typing as at
import tmrl_openpi.shared.nnx_utils as nnx_utils
from tmrl_openpi.models.pi0 import Pi0, make_attn_mask
from tmrl_openpi.models.pi0_config import Pi0Config

logger = logging.getLogger("tmrl_openpi")


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@at.typecheck
def posemb_dual_sincos(
    pos: at.Real[at.Array, " b"],
    pos_prefix: at.Real[at.Array, " b"],
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes dual sine-cosine positional embeddings for two scalar positions."""
    if embedding_dim % 4 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 4")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 4)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    sinusoid_input_prefix = jnp.einsum(
        "i,j->ij",
        pos_prefix,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate(
        [
            jnp.sin(sinusoid_input),
            jnp.cos(sinusoid_input),
            jnp.sin(sinusoid_input_prefix),
            jnp.cos(sinusoid_input_prefix),
        ],
        axis=-1,
    )


@dataclasses.dataclass(frozen=True)
class TMPi0Config(Pi0Config):
    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.TMPi05 if self.pi05 else _model.ModelType.TMPi0

    @override
    def create(self, rng) -> "TMPi0":
        return TMPi0(self, rngs=nnx.Rngs(rng))


class TMPi0(Pi0):
    def __init__(self, config: TMPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.T = 1000
        betas = np.linspace(1e-4, 0.02, self.T, dtype=np.float32)
        alphas = 1.0 - betas
        # Store as Python list (not a JAX array) to avoid NNX treating it as state.
        self.alpha_bars = np.cumprod(alphas, axis=0, dtype=np.float32).tolist()

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        timestep_prefix: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_dual_sincos(
            timestep, timestep_prefix, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, noise_prefix_rng, time_prefix_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # forward pass of prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # noise the prefix using a DDIM-style marginal controlled by time_prefix
        time_prefix = jax.random.uniform(time_prefix_rng, batch_shape)
        noise_prefix = jax.random.normal(noise_prefix_rng, prefix_tokens.shape)
        t_idx = jnp.clip((time_prefix * (self.T - 1)).astype(jnp.int32), 0, self.T - 1)
        ab_t = jnp.asarray(self.alpha_bars, dtype=jnp.float32)[t_idx]
        sqrt_ab = jnp.sqrt(ab_t)[..., None, None]
        sqrt_bb = jnp.sqrt(1.0 - ab_t)[..., None, None]
        noise_prefix = noise_prefix.astype(prefix_tokens.dtype)
        sqrt_ab = sqrt_ab.astype(prefix_tokens.dtype)
        sqrt_bb = sqrt_bb.astype(prefix_tokens.dtype)
        noisy_prefix_tokens = sqrt_ab * prefix_tokens + sqrt_bb * noise_prefix

        # forward pass of suffix
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time, time_prefix)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [noisy_prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        observation: _model.Observation,
        *,
        noise: jnp.ndarray,
        noise_prefix: jnp.ndarray | None = None,
        time_prefix: jnp.ndarray | None = None,
        rng: at.KeyArrayLike | None = None,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # jax.debug.print('time_prefix: {}', time_prefix)

        if time_prefix is None:
            time_prefix = jnp.zeros(batch_size, dtype=jnp.float32)
        else:
            time_prefix = jnp.broadcast_to(jnp.clip(time_prefix, 0.0, 1.0), (batch_size,))
            time_prefix = jnp.asarray(time_prefix, dtype=jnp.float32)

        # if jnp.all(time_prefix == 0):
        #     noisy_prefix_tokens = prefix_tokens
        # else:
        # jax.debug.print('noise_prefix: {}', noise_prefix)
        # jax.debug.print('noise: {}', noise)
        if noise_prefix is None:
            if rng is None:
                raise ValueError("Provide `rng` when supplying non-zero `time_prefix` without `noise_prefix`.")
            rng, noise_prefix_rng = jax.random.split(rng)
            noise_prefix = jax.random.normal(noise_prefix_rng, prefix_tokens.shape)
        t_idx = jnp.clip((time_prefix * (self.T - 1)).astype(jnp.int32), 0, self.T - 1)
        ab_t = jnp.asarray(self.alpha_bars, dtype=jnp.float32)[t_idx]
        sqrt_ab = jnp.sqrt(ab_t)[..., None, None]
        sqrt_bb = jnp.sqrt(1.0 - ab_t)[..., None, None]
        noise_prefix = noise_prefix.astype(prefix_tokens.dtype)
        sqrt_ab = sqrt_ab.astype(prefix_tokens.dtype)
        sqrt_bb = sqrt_bb.astype(prefix_tokens.dtype)
        noisy_prefix_tokens = sqrt_ab * prefix_tokens + sqrt_bb * noise_prefix

        noisy_norm = jnp.linalg.norm(noisy_prefix_tokens, axis=-1, keepdims=True)
        clean_norm = jnp.linalg.norm(prefix_tokens, axis=-1, keepdims=True)
        cosine_sim = jnp.sum(noisy_prefix_tokens * prefix_tokens, axis=-1) / (
            noisy_norm.squeeze(-1) * clean_norm.squeeze(-1) + 1e-8
        )
        # jax.debug.print('cosine_sim (mean): {}', jnp.mean(cosine_sim))

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([noisy_prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), time_prefix
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
