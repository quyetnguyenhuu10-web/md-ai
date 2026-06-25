"""Normalization layers used across modalities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class RMSNorm(nn.Module):
    with_scale: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + 1.0e-6)

        if self.with_scale:
            scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
            scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
            x = x * scale
        return x
