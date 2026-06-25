"""Activation / feed-forward primitives used across modalities."""

from __future__ import annotations

import jax
from flax import linen as nn

from mintdim_lab.model.layers.linear import kernel_init

_SUPPORTED_ACTIVATIONS = frozenset(
    {
        "gelu",
        "silu",
        "geglu",
        "swiglu",
    }
)


class FeedForward(nn.Module):
    features: int
    hidden_dim: int
    activation: str = "gelu"

    def _activation_name(self) -> str:
        activation = str(self.activation).strip().lower()

        if activation not in _SUPPORTED_ACTIVATIONS:
            supported = ", ".join(sorted(_SUPPORTED_ACTIVATIONS))
            raise ValueError(
                f"Unsupported feed-forward activation: {self.activation!r}. "
                f"Expected one of: {supported}."
            )

        return activation

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        activation = self._activation_name()

        up = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=kernel_init(),
            name="up_proj",
        )(x)

        if activation == "gelu":
            hidden = jax.nn.gelu(up, approximate=True)

        elif activation == "silu":
            hidden = jax.nn.silu(up)

        elif activation == "geglu":
            gate = nn.Dense(
                self.hidden_dim,
                use_bias=False,
                kernel_init=kernel_init(),
                name="gate_proj",
            )(x)
            hidden = jax.nn.gelu(gate, approximate=True) * up

        elif activation == "swiglu":
            gate = nn.Dense(
                self.hidden_dim,
                use_bias=False,
                kernel_init=kernel_init(),
                name="gate_proj",
            )(x)
            hidden = jax.nn.silu(gate) * up

        else:
            # This branch is unreachable because _activation_name validates first.
            raise AssertionError(f"Unhandled feed-forward activation: {activation!r}")

        return nn.Dense(
            self.features,
            use_bias=False,
            kernel_init=kernel_init(),
            name="down_proj",
        )(hidden)
