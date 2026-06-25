from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from mintdim_lab.model.layers import RMSNorm, kernel_init

from .config import AttentionType, TransformerConfig
from .rope import build_position_qk

_FLASH_ATTENTION_BLOCK_SIZE = 64


class SelfAttention(nn.Module):
    """Self-attention with explicit backend contracts.

    ``use_flash_attention`` selects MintDim's custom blockwise FlashAttention
    path. It is implemented with portable JAX/XLA primitives and is intended to
    be backend portable: CPU, GPU, and TPU runtimes may use it when their
    JAX/XLA backend can compile the primitive operations used here.

    ``use_fused_attention`` selects JAX's cuDNN implementation by passing
    ``implementation="cudnn"`` to ``jax.nn.dot_product_attention``. This is
    NVIDIA GPU/cuDNN flash attention only; it is not a TPU backend.
    """

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
        attn_type = cfg.attention_types[self.layer_id]
        rope_base = cfg.rope_base_for_layer(self.layer_id)
        rope_scale = cfg.rope_scale_for_layer(self.layer_id)
        rope_dim = cfg.rope_dim_for_layer(self.layer_id)
        qk_logits_softcap = cfg.qk_logits_softcap_for_layer(self.layer_id)
        kv_heads = cfg.kv_heads_for_layer(self.layer_id)
        key_size = cfg.key_size_for_layer(self.layer_id)
        value_size = cfg.value_size_for_layer(self.layer_id)

        q = nn.Dense(
            cfg.num_heads * key_size,
            use_bias=False,
            kernel_init=kernel_init(),
            name="q_proj",
        )(x)
        k = nn.Dense(
            kv_heads * key_size,
            use_bias=False,
            kernel_init=kernel_init(),
            name="k_proj",
        )(x)
        v = nn.Dense(
            kv_heads * value_size,
            use_bias=False,
            kernel_init=kernel_init(),
            name="v_proj",
        )(x)

        batch_size, seq_len, _ = x.shape
        q = q.reshape(batch_size, seq_len, cfg.num_heads, key_size)
        k = k.reshape(batch_size, seq_len, kv_heads, key_size)
        v = v.reshape(batch_size, seq_len, kv_heads, value_size)

        if cfg.qk_norm_enabled:
            q = RMSNorm(with_scale=cfg.qk_norm_with_scale, name="q_norm")(q)
            k = RMSNorm(with_scale=cfg.qk_norm_with_scale, name="k_norm")(k)

        new_cache = None
        key_positions = None
        valid_key_count = None
        if cache is not None:
            if cache_index is None:
                cache_index = jnp.array(0, dtype=jnp.int32)
            cache_index = jnp.asarray(cache_index, dtype=jnp.int32)
            key_cache = jnp.asarray(cache["key"])
            value_cache = jnp.asarray(cache["value"])
            k = jax.lax.dynamic_update_slice(key_cache, k, (0, cache_index, 0, 0))
            v = jax.lax.dynamic_update_slice(value_cache, v, (0, cache_index, 0, 0))
            new_cache = {"key": k, "value": v}
            max_key_len = k.shape[1]
            key_positions = jnp.broadcast_to(
                jnp.arange(max_key_len, dtype=jnp.int32)[None, :],
                (batch_size, max_key_len),
            )
            valid_key_count = cache_index + jnp.asarray(seq_len, dtype=jnp.int32)

        repeat = cfg.num_heads // kv_heads
        if repeat > 1:
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        attention_key_positions = key_positions if key_positions is not None else positions
        q_dot, k_dot = build_position_qk(
            q,
            k,
            positions,
            attention_key_positions,
            rope_dim=rope_dim,
            base_frequency=rope_base,
            scale_factor=rope_scale,
        )
        scale = float(1.0 / (float(key_size) ** 0.5))
        if cfg.use_flash_attention:
            out = blockwise_flash_attention(
                q_dot,
                k_dot,
                v,
                positions=positions,
                attn_type=attn_type,
                sliding_window_size=cfg.sliding_window_size,
                key_positions=attention_key_positions,
                valid_key_count=valid_key_count,
                scale=scale,
                softcap=qk_logits_softcap,
            )
        elif cfg.use_fused_attention and _can_use_fused_attention(q_dot, v, qk_logits_softcap):
            mask = make_attention_mask(
                positions,
                attn_type=attn_type,
                sliding_window_size=cfg.sliding_window_size,
                key_positions=attention_key_positions,
                valid_key_count=valid_key_count,
            )
            # ``use_fused_attention`` deliberately means NVIDIA GPU/cuDNN
            # flash attention. If the current backend or shape is not accepted,
            # this branch is skipped silently and the eager path below keeps the
            # full four-branch attention semantics.
            out = jax.nn.dot_product_attention(
                q_dot,
                k_dot,
                v,
                mask=mask.astype(jnp.bool_),
                scale=scale,
                implementation="cudnn",
            )
        else:
            mask = make_attention_mask(
                positions,
                attn_type=attn_type,
                sliding_window_size=cfg.sliding_window_size,
                key_positions=attention_key_positions,
                valid_key_count=valid_key_count,
            )
            scores = jnp.einsum(
                "bqhd,bkhd->bhqk",
                q_dot,
                k_dot,
                preferred_element_type=jnp.float32,
            ).astype(jnp.float32)
            scores = scores * jax.lax.rsqrt(jnp.array(key_size, dtype=jnp.float32))
            if qk_logits_softcap is not None:
                cap = jnp.array(float(qk_logits_softcap), dtype=scores.dtype)
                scores = jnp.tanh(scores / cap) * cap
            scores = jnp.where(mask, scores, jnp.array(-1.0e30, dtype=scores.dtype))
            probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
            out = jnp.einsum(
                "bhqk,bkhd->bqhd",
                probs,
                v,
                preferred_element_type=x.dtype,
            )
        out = out.reshape(batch_size, seq_len, cfg.num_heads * value_size)
        return nn.Dense(
            cfg.embed_dim,
            use_bias=False,
            kernel_init=kernel_init(),
            name="o_proj",
        )(out), new_cache


def make_attention_mask(
    positions: jax.Array,
    *,
    attn_type: AttentionType,
    sliding_window_size: int,
    key_positions: jax.Array | None = None,
    valid_key_count: jax.Array | int | None = None,
) -> jax.Array:
    query_pos = positions[:, :, None]
    if key_positions is None:
        key_positions = positions
    key_pos = key_positions[:, None, :]

    mask = key_pos <= query_pos

    if valid_key_count is not None:
        mask = jnp.logical_and(
            mask,
            key_pos < jnp.asarray(valid_key_count, dtype=jnp.int32),
        )

    if attn_type == AttentionType.LOCAL_SLIDING:
        mask = jnp.logical_and(
            mask,
            key_pos > query_pos - int(sliding_window_size),
        )

    return mask[:, None, :, :]


def _can_use_fused_attention(
    q: jax.Array,
    v: jax.Array,
    qk_logits_softcap: float | None,
) -> bool:
    if qk_logits_softcap is not None:
        return False
    if jax.default_backend() != "gpu":
        return False
    return int(q.shape[-1]) == int(v.shape[-1])


def blockwise_flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    positions: jax.Array,
    attn_type: AttentionType,
    sliding_window_size: int,
    key_positions: jax.Array | None = None,
    valid_key_count: jax.Array | int | None = None,
    scale: float,
    softcap: float | None = None,
    block_size: int = _FLASH_ATTENTION_BLOCK_SIZE,
) -> jax.Array:
    """Portable custom FlashAttention via blockwise online softmax.

    This keeps the semantics of the eager attention path but scans over
    key/value blocks, so it never materializes full float32 ``Q x K`` scores or
    probabilities.

    This implementation is built from ordinary JAX/XLA primitives rather than
    cuDNN, CUDA-specific kernels, or Pallas. It is intended to be backend
    portable across CPU, GPU, and TPU. It may run on any runtime that can compile those primitives.
    Numerical results are
    equivalent to eager attention up to normal floating-point rounding.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if key_positions is None:
        key_positions = positions

    batch_size, query_len, num_heads, _key_size = q.shape
    key_len = k.shape[1]
    value_size = v.shape[-1]
    pad_len = (-int(key_len)) % int(block_size)

    k = _pad_kv_sequence(k, pad_len)
    v = _pad_kv_sequence(v, pad_len)
    key_positions = _pad_key_positions(key_positions, pad_len)

    padded_key_len = k.shape[1]
    num_blocks = padded_key_len // int(block_size)
    qh = jnp.transpose(q.astype(jnp.float32), (0, 2, 1, 3))
    kh = jnp.transpose(
        k.reshape(batch_size, num_blocks, int(block_size), num_heads, _key_size),
        (1, 0, 3, 2, 4),
    )
    vh = jnp.transpose(
        v.reshape(batch_size, num_blocks, int(block_size), num_heads, value_size),
        (1, 0, 3, 2, 4),
    )
    key_pos_blocks = jnp.transpose(
        key_positions.reshape(batch_size, num_blocks, int(block_size)),
        (1, 0, 2),
    )

    neg_inf = jnp.array(-jnp.inf, dtype=jnp.float32)
    m0 = jnp.full((batch_size, num_heads, query_len), neg_inf, dtype=jnp.float32)
    l0 = jnp.zeros((batch_size, num_heads, query_len), dtype=jnp.float32)
    out0 = jnp.zeros((batch_size, num_heads, query_len, value_size), dtype=jnp.float32)

    def scan_block(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        block: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        m_prev, l_prev, out_prev = carry
        k_block, v_block, key_pos_block = block
        scores = jnp.einsum(
            "bhqd,bhkd->bhqk",
            qh,
            k_block,
            preferred_element_type=jnp.float32,
        )
        scores = scores * jnp.asarray(scale, dtype=jnp.float32)
        if softcap is not None:
            cap = jnp.array(float(softcap), dtype=jnp.float32)
            scores = jnp.tanh(scores / cap) * cap

        block_mask = _attention_mask_block(
            positions,
            key_pos_block,
            attn_type=attn_type,
            sliding_window_size=sliding_window_size,
            valid_key_count=valid_key_count,
        )
        scores = jnp.where(block_mask, scores, neg_inf)

        block_max = jnp.max(scores, axis=-1)
        m_next = jnp.maximum(m_prev, block_max)
        m_next_safe = jnp.where(jnp.isfinite(m_next), m_next, 0.0)
        m_prev_safe = jnp.where(jnp.isfinite(m_prev), m_prev, 0.0)
        prev_scale = jnp.where(jnp.isfinite(m_prev), jnp.exp(m_prev_safe - m_next_safe), 0.0)
        probs = jnp.where(
            jnp.isfinite(scores),
            jnp.exp(scores - m_next_safe[..., None]),
            0.0,
        )
        l_next = l_prev * prev_scale + jnp.sum(probs, axis=-1)
        out_next = out_prev * prev_scale[..., None] + jnp.einsum(
            "bhqk,bhkd->bhqd",
            probs,
            v_block.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        return (m_next, l_next, out_next), None

    (_m, normalizer, out), _ = jax.lax.scan(scan_block, (m0, l0, out0), (kh, vh, key_pos_blocks))
    out = jnp.where(normalizer[..., None] > 0.0, out / normalizer[..., None], 0.0)
    return jnp.transpose(out, (0, 2, 1, 3)).astype(q.dtype)


def _pad_kv_sequence(value: jax.Array, pad_len: int) -> jax.Array:
    if int(pad_len) == 0:
        return value
    return jnp.pad(
        value,
        ((0, 0), (0, int(pad_len)), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def _pad_key_positions(key_positions: jax.Array, pad_len: int) -> jax.Array:
    if int(pad_len) == 0:
        return key_positions
    return jnp.pad(
        key_positions,
        ((0, 0), (0, int(pad_len))),
        mode="constant",
        constant_values=jnp.iinfo(jnp.int32).max,
    )


def _attention_mask_block(
    positions: jax.Array,
    key_positions: jax.Array,
    *,
    attn_type: AttentionType,
    sliding_window_size: int,
    valid_key_count: jax.Array | int | None,
) -> jax.Array:
    query_pos = positions[:, :, None]
    key_pos = key_positions[:, None, :]
    mask = key_pos <= query_pos

    if valid_key_count is not None:
        valid = jnp.asarray(valid_key_count, dtype=jnp.int32)
        if valid.ndim == 1:
            valid = valid[:, None, None]
        mask = jnp.logical_and(mask, key_pos < valid)

    if attn_type == AttentionType.LOCAL_SLIDING:
        mask = jnp.logical_and(
            mask,
            key_pos > query_pos - int(sliding_window_size),
        )

    return mask[:, None, :, :]
