"""Linear / einsum primitives used across modalities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


def kernel_init() -> nn.initializers.Initializer:
    return nn.initializers.normal(stddev=0.02)


def _unwrap_param(value):
    if isinstance(value, dict) and "w" in value:
        return value["w"]
    return value


class Einsum(nn.Module):
    shape: tuple[int, ...]
    weight_name: str = "w"
    initializer: nn.initializers.Initializer = nn.initializers.normal()
    dtype: jnp.dtype | None = None
    w_scale: float | None = None

    @nn.compact
    def __call__(self, equation: str, x: jax.Array) -> jax.Array:
        weight = self.param(
            self.weight_name,
            self.initializer,
            self.shape,
            self.dtype if self.dtype is not None else None,
        )
        weight = _unwrap_param(weight)
        if self.w_scale is not None:
            weight *= self.w_scale
        return jnp.einsum(equation, x, weight)


class ClippedEinsum(nn.Module):
    shape: tuple[int, ...]
    weight_name: str = "w"
    initializer: nn.initializers.Initializer = nn.initializers.normal()
    dtype: jnp.dtype | None = None
    w_scale: float | None = None

    @nn.compact
    def __call__(self, equation: str, x: jax.Array) -> jax.Array:
        weight = self.param(
            self.weight_name,
            self.initializer,
            self.shape,
            self.dtype if self.dtype is not None else None,
        )
        weight = _unwrap_param(weight)
        if self.w_scale is not None:
            weight *= self.w_scale

        clip_input_min = self.param(
            "clip_input_min",
            lambda key, shape, dtype=None: jnp.array(float("-inf")),
            (),
        )
        clip_input_max = self.param(
            "clip_input_max",
            lambda key, shape, dtype=None: jnp.array(float("inf")),
            (),
        )
        clip_output_min = self.param(
            "clip_output_min",
            lambda key, shape, dtype=None: jnp.array(float("-inf")),
            (),
        )
        clip_output_max = self.param(
            "clip_output_max",
            lambda key, shape, dtype=None: jnp.array(float("inf")),
            (),
        )

        x = jnp.clip(x, clip_input_min, clip_input_max)
        x = jnp.einsum(equation, x, weight)
        return jnp.clip(x, clip_output_min, clip_output_max)
