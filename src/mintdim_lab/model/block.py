from __future__ import annotations

import jax
from flax import linen as nn

from mintdim_lab.model.layers import FeedForward, RMSNorm

from .attention import SelfAttention
from .config import TransformerConfig


class TransformerBlock(nn.Module):
    config: TransformerConfig
    layer_id: int

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        *,
        cache: dict[str, jax.Array] | None = None,
        cache_index: jax.Array | int | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array] | None]:
        cfg = self.config
        residual = x
        h = RMSNorm(name="pre_attention_norm")(x)
        attn_out, new_cache = SelfAttention(cfg, self.layer_id, name="attn")(
            h,
            positions,
            cache=cache,
            cache_index=cache_index,
        )
        if cfg.use_post_attn_norm:
            attn_out = RMSNorm(name="post_attention_norm")(attn_out)
        attn_scale = self.param("attn_scale", nn.initializers.ones, (cfg.embed_dim,))
        x = residual + attn_out * attn_scale.astype(attn_out.dtype)

        ffw = RMSNorm(name="pre_ffw_norm")(x)
        ffw = FeedForward(
            features=cfg.embed_dim,
            hidden_dim=cfg.dense_hidden_dim,
            activation=cfg.ffw_activation,
            name="mlp",
        )(ffw)
        if cfg.use_post_ffw_norm:
            ffw = RMSNorm(name="post_ffw_norm")(ffw)

        ffw_scale = self.param("ffw_scale", nn.initializers.ones, (cfg.embed_dim,))
        out = x + ffw * ffw_scale.astype(ffw.dtype)
        return out, new_cache
