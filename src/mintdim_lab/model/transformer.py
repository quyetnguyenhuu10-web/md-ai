"""Top-level text Transformer for MintDim.

This merges the previous ``transformer.py`` (the :class:`Transformer` module)
and ``decoder.py`` (the ``run_decoder`` helper) into a single file. The decoder
loop is intentionally inlined here because it has only one caller.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from mintdim_lab.model.layers import RMSNorm

from .block import TransformerBlock
from .config import TransformerConfig
from .embedding import (
    compute_logits,
    create_input_embedding,
    create_output_head,
    embed_tokens,
)


@struct.dataclass
class TransformerOutput:
    logits: jax.Array
    cache: Any | None = None
    hidden_states: jax.Array | None = None


class Transformer(nn.Module):
    """Training-oriented MintDim decoder."""

    config: TransformerConfig

    def setup(self) -> None:
        self.config.validate()

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        *,
        positions: jax.Array | None = None,
        cache: Any | None = None,
        cache_index: jax.Array | int | None = None,
        return_cache: bool = False,
        return_hidden: bool = False,
    ) -> jax.Array | TransformerOutput:
        cfg = self.config
        # tokens.shape[-1] is the actual sequence length supplied by the
        # caller. For unit_read training this is the .bin unit length, not
        # cfg.max_seq_len. max_seq_len is only the model context ceiling.
        seq_len = tokens.shape[-1]

        if seq_len > cfg.max_seq_len:
            raise ValueError(f"seq_len must be <= max_seq_len ({cfg.max_seq_len})")

        if positions is None:
            base_position = (
                jnp.array(0, dtype=jnp.int32)
                if cache_index is None
                else jnp.asarray(cache_index, dtype=jnp.int32)
            )
            positions = jnp.broadcast_to(
                (base_position + jnp.arange(seq_len, dtype=jnp.int32))[None, :],
                tokens.shape,
            )

        embedding = create_input_embedding(
            self,
            num_embed=cfg.num_embed,
            embed_dim=cfg.embed_dim,
        )

        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        x = embed_tokens(
            embedding,
            tokens,
            embed_dim=cfg.embed_dim,
        )
        new_cache_layers = [] if cache is not None else None
        block_cls = TransformerBlock
        if cfg.use_gradient_checkpointing:
            block_cls = nn.remat(TransformerBlock, prevent_cse=False)
        for layer_id in range(cfg.num_layers):
            layer_cache = cache[layer_id] if cache is not None else None
            x, layer_new_cache = block_cls(
                cfg,
                layer_id,
                name=f"layer_{layer_id}",
            )(
                x,
                positions,
                cache=layer_cache,
                cache_index=cache_index,
            )
            if new_cache_layers is not None:
                new_cache_layers.append(layer_new_cache)

        x = RMSNorm(name="final_norm")(x)
        output_cache = tuple(new_cache_layers) if new_cache_layers is not None else None

        output_head = create_output_head(
            self,
            input_embedding=embedding,
            logits_head=cfg.logits_head,
            num_embed=cfg.num_embed,
            embed_dim=cfg.embed_dim,
        )

        logits = compute_logits(
            x,
            output_head,
            final_logit_softcap=cfg.final_logit_softcap,
        )

        if not return_cache and not return_hidden:
            return logits

        hidden_states = x if return_hidden else None

        return TransformerOutput(
            logits=logits,
            cache=output_cache,
            hidden_states=hidden_states,
        )
