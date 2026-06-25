from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from mintdim_lab.model.layers import kernel_init


def create_input_embedding(
    module: nn.Module,
    *,
    num_embed: int,
    embed_dim: int,
) -> jax.Array:
    embedding = module.param(
        "input_embedding",
        kernel_init(),
        (int(num_embed), int(embed_dim)),
    )
    return jnp.asarray(embedding)


def embed_tokens(
    embedding: jax.Array,
    tokens: jax.Array,
    *,
    embed_dim: int,
) -> jax.Array:
    x = jnp.take(
        jnp.asarray(embedding),
        jnp.asarray(tokens, dtype=jnp.int32),
        axis=0,
    )
    return x * jnp.sqrt(jnp.array(int(embed_dim), dtype=x.dtype))


def create_output_head(
    module: nn.Module,
    *,
    input_embedding: jax.Array,
    logits_head: str,
    num_embed: int,
    embed_dim: int,
) -> jax.Array:
    if str(logits_head) == "tied":
        return jnp.asarray(input_embedding)

    output_head = module.param(
        "output_head",
        kernel_init(),
        (int(num_embed), int(embed_dim)),
    )
    return jnp.asarray(output_head)


def compute_logits(
    hidden_states: jax.Array,
    output_head: jax.Array,
    *,
    final_logit_softcap: float | None,
) -> jax.Array:
    logits = jnp.einsum(
        "bld,vd->blv",
        hidden_states,
        jnp.asarray(output_head, dtype=hidden_states.dtype),
        preferred_element_type=hidden_states.dtype,
    )
    if final_logit_softcap is not None:
        cap = jnp.array(float(final_logit_softcap), dtype=logits.dtype)
        logits = jnp.tanh(logits / cap) * cap
    return logits
