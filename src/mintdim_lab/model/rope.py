from __future__ import annotations

import jax
import jax.numpy as jnp


def apply_rope_attention_logits(
    q: jax.Array,
    k: jax.Array,
    q_positions: jax.Array,
    k_positions: jax.Array | None = None,
    *,
    base_frequency: int,
    scale_factor: float,
) -> jax.Array:
    """Compute pure RoPE relative-position logits on an even-dimensional slice."""
    if k_positions is None:
        k_positions = q_positions

    head_dim = int(q.shape[-1])
    if head_dim <= 0:
        batch_size, q_len, heads, _ = q.shape
        k_len = k.shape[1]
        return jnp.zeros((batch_size, heads, q_len, k_len), dtype=q.dtype)
    if head_dim % 2 != 0:
        raise ValueError("RoPE head dim must be even")

    inv_freq = build_rope_inv_freq(
        head_dim=head_dim,
        base_frequency=base_frequency,
    )
    relative_positions = (k_positions[:, None, :] - q_positions[:, :, None]).astype(jnp.float32)

    angles = (relative_positions[..., None] / float(scale_factor)) * inv_freq
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    q_even = q[..., 0::2].astype(jnp.float32)
    q_odd = q[..., 1::2].astype(jnp.float32)
    k_even = k[..., 0::2].astype(jnp.float32)
    k_odd = k[..., 1::2].astype(jnp.float32)

    rotated_k_even = (
        k_even[:, None, :, :, :] * cos[:, :, :, None, :]
        - k_odd[:, None, :, :, :] * sin[:, :, :, None, :]
    )
    rotated_k_odd = (
        k_even[:, None, :, :, :] * sin[:, :, :, None, :]
        + k_odd[:, None, :, :, :] * cos[:, :, :, None, :]
    )
    logits = jnp.sum(
        q_even[:, :, None, :, :] * rotated_k_even + q_odd[:, :, None, :, :] * rotated_k_odd,
        axis=-1,
    )

    return jnp.transpose(logits, (0, 3, 1, 2)).astype(q.dtype)


def apply_nope_attention_logits(q: jax.Array, k: jax.Array) -> jax.Array:
    """Compute pure content dot-product logits."""
    if int(q.shape[-1]) <= 0:
        batch_size, q_len, heads, _ = q.shape
        k_len = k.shape[1]
        return jnp.zeros((batch_size, heads, q_len, k_len), dtype=q.dtype)

    return jnp.einsum(
        "bqhd,bkhd->bhqk",
        q,
        k,
        preferred_element_type=jnp.float32,
    ).astype(q.dtype)


def build_position_qk(
    q: jax.Array,
    k: jax.Array,
    q_positions: jax.Array,
    k_positions: jax.Array | None = None,
    *,
    rope_dim: int,
    base_frequency: int,
    scale_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """Convert split RoPE/NoPE Q/K into dot-product features.

    The leading ``rope_dim`` features receive RoPE and the remaining features
    stay as NoPE content channels. The returned ``q_dot @ k_dot`` equals the
    desired split logits, so dot-product attention backends can consume it
    unchanged.
    """
    if k_positions is None:
        k_positions = q_positions

    head_dim = int(q.shape[-1])
    rope_dim = int(rope_dim)
    if rope_dim < 0:
        raise ValueError("rope_dim must be >= 0")
    if rope_dim > head_dim:
        raise ValueError("rope_dim must be <= Q/K head dim")
    if rope_dim > 0 and rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even when positive")

    if rope_dim == 0:
        return q, k

    if rope_dim == head_dim:
        q_rope = apply_rope_to_vector(
            q,
            q_positions,
            base_frequency=base_frequency,
            scale_factor=scale_factor,
        )
        k_rope = apply_rope_to_vector(
            k,
            k_positions,
            base_frequency=base_frequency,
            scale_factor=scale_factor,
        )
        return q_rope, k_rope

    q_rope = apply_rope_to_vector(
        q[..., :rope_dim],
        q_positions,
        base_frequency=base_frequency,
        scale_factor=scale_factor,
    )
    k_rope = apply_rope_to_vector(
        k[..., :rope_dim],
        k_positions,
        base_frequency=base_frequency,
        scale_factor=scale_factor,
    )
    return (
        jnp.concatenate((q_rope, q[..., rope_dim:]), axis=-1),
        jnp.concatenate((k_rope, k[..., rope_dim:]), axis=-1),
    )


def apply_split_rope_nope_attention_logits(
    q: jax.Array,
    k: jax.Array,
    q_positions: jax.Array,
    k_positions: jax.Array | None = None,
    *,
    rope_dim: int,
    base_frequency: int,
    scale_factor: float,
) -> jax.Array:
    """Compute attention logits for a fixed RoPE/NoPE split."""
    q_dot, k_dot = build_position_qk(
        q,
        k,
        q_positions,
        k_positions,
        rope_dim=rope_dim,
        base_frequency=base_frequency,
        scale_factor=scale_factor,
    )
    return apply_nope_attention_logits(q_dot, k_dot)


def apply_rope_to_vector(
    x: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    scale_factor: float,
) -> jax.Array:
    """Apply standard absolute RoPE to an even-dimensional Q/K vector slice."""
    head_dim = int(x.shape[-1])
    if head_dim <= 0:
        return x
    if head_dim % 2 != 0:
        raise ValueError("RoPE head dim must be even")

    inv_freq = build_rope_inv_freq(
        head_dim=head_dim,
        base_frequency=base_frequency,
    )
    angles = (positions.astype(jnp.float32)[..., None] / float(scale_factor)) * inv_freq
    cos = jnp.cos(angles)[:, :, None, :]
    sin = jnp.sin(angles)[:, :, None, :]

    x_even = x[..., 0::2].astype(jnp.float32)
    x_odd = x[..., 1::2].astype(jnp.float32)
    rotated = jnp.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        axis=-1,
    ).reshape(x.shape)

    return rotated.astype(x.dtype)


def build_rope_inv_freq(
    *,
    head_dim: int,
    base_frequency: int,
) -> jax.Array:
    """Build standard RoPE inverse frequencies for even/odd pairs."""
    return 1.0 / (
        float(base_frequency)
        ** (jnp.arange(0, int(head_dim), 2, dtype=jnp.float32) / float(head_dim))
    )
