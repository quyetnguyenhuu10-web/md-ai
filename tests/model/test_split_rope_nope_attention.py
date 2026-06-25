from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from test_nn_text_only import make_small_config

from mintdim_lab.model import Transformer
from mintdim_lab.model.rope import (
    apply_nope_attention_logits,
    apply_rope_attention_logits,
    apply_split_rope_nope_attention_logits,
    build_position_qk,
)


def test_split_rope_dim_zero_is_content_dot_product():
    q = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 1, 4))
    k = jax.random.normal(jax.random.PRNGKey(1), (1, 3, 1, 4))
    q_positions = jnp.asarray([[0, 2]], dtype=jnp.int32)
    k_positions = jnp.asarray([[0, 1, 3]], dtype=jnp.int32)

    logits = apply_split_rope_nope_attention_logits(
        q,
        k,
        q_positions,
        k_positions,
        rope_dim=0,
        base_frequency=10000,
        scale_factor=1.0,
    )
    expected = apply_nope_attention_logits(q, k)

    assert jnp.allclose(logits, expected, atol=1.0e-6)


def test_split_rope_full_dim_is_pure_rope():
    q = jax.random.normal(jax.random.PRNGKey(2), (1, 2, 1, 4))
    k = jax.random.normal(jax.random.PRNGKey(3), (1, 3, 1, 4))
    q_positions = jnp.asarray([[0, 2]], dtype=jnp.int32)
    k_positions = jnp.asarray([[0, 1, 3]], dtype=jnp.int32)

    logits = apply_split_rope_nope_attention_logits(
        q,
        k,
        q_positions,
        k_positions,
        rope_dim=4,
        base_frequency=10000,
        scale_factor=1.0,
    )
    expected = apply_rope_attention_logits(
        q,
        k,
        q_positions,
        k_positions,
        base_frequency=10000,
        scale_factor=1.0,
    )

    assert jnp.allclose(logits, expected, atol=1.0e-5)


def test_split_rope_nope_logits_are_sum_of_slices():
    q = jax.random.normal(jax.random.PRNGKey(4), (1, 3, 2, 8))
    k = jax.random.normal(jax.random.PRNGKey(5), (1, 4, 2, 8))
    q_positions = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    k_positions = jnp.asarray([[0, 2, 3, 5]], dtype=jnp.int32)

    logits = apply_split_rope_nope_attention_logits(
        q,
        k,
        q_positions,
        k_positions,
        rope_dim=4,
        base_frequency=10000,
        scale_factor=1.0,
    ).astype(jnp.float32)
    rope = apply_rope_attention_logits(
        q[..., :4],
        k[..., :4],
        q_positions,
        k_positions,
        base_frequency=10000,
        scale_factor=1.0,
    ).astype(jnp.float32)
    nope = apply_nope_attention_logits(q[..., 4:], k[..., 4:]).astype(jnp.float32)

    assert jnp.allclose(logits, rope + nope, atol=1.0e-5)


def test_split_rope_nope_logits_are_shift_invariant():
    q = jax.random.normal(jax.random.PRNGKey(6), (1, 3, 2, 8))
    k = jax.random.normal(jax.random.PRNGKey(7), (1, 4, 2, 8))
    q_positions = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    k_positions = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)

    base = apply_split_rope_nope_attention_logits(
        q,
        k,
        q_positions,
        k_positions,
        rope_dim=4,
        base_frequency=10000,
        scale_factor=1.0,
    )
    shifted = apply_split_rope_nope_attention_logits(
        q,
        k,
        q_positions + 1000,
        k_positions + 1000,
        rope_dim=4,
        base_frequency=10000,
        scale_factor=1.0,
    )

    assert jnp.allclose(base, shifted, atol=1.0e-4)


def test_build_position_qk_dot_matches_split_logits():
    q = jax.random.normal(jax.random.PRNGKey(8), (1, 3, 2, 8))
    k = jax.random.normal(jax.random.PRNGKey(9), (1, 4, 2, 8))
    q_positions = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    k_positions = jnp.asarray([[0, 2, 3, 5]], dtype=jnp.int32)

    q_dot, k_dot = build_position_qk(
        q,
        k,
        q_positions,
        k_positions,
        rope_dim=4,
        base_frequency=10000,
        scale_factor=1.0,
    )
    actual = apply_nope_attention_logits(q_dot, k_dot).astype(jnp.float32)
    expected = apply_split_rope_nope_attention_logits(
        q,
        k,
        q_positions,
        k_positions,
        rope_dim=4,
        base_frequency=10000,
        scale_factor=1.0,
    ).astype(jnp.float32)

    assert q_dot.shape[-1] == q.shape[-1]
    assert k_dot.shape[-1] == k.shape[-1]
    assert jnp.allclose(actual, expected, atol=1.0e-5)


@pytest.mark.parametrize("rope_dim", [-1, 5, 10])
def test_build_position_qk_rejects_invalid_rope_dim(rope_dim):
    q = jnp.zeros((1, 2, 2, 8), dtype=jnp.float32)
    k = jnp.zeros((1, 2, 2, 8), dtype=jnp.float32)
    positions = jnp.asarray([[0, 1]], dtype=jnp.int32)

    with pytest.raises(ValueError, match="rope_dim"):
        build_position_qk(
            q,
            k,
            positions,
            positions,
            rope_dim=rope_dim,
            base_frequency=10000,
            scale_factor=1.0,
        )


def test_attention_forward_runs_with_split_rope_nope():
    cfg = make_small_config()
    model = Transformer(cfg)

    tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    variables = model.init(jax.random.PRNGKey(0), tokens)
    logits = model.apply(variables, tokens)

    assert "rope_gate_logit" not in variables["params"]["layer_0"]["attn"]
    assert logits.shape == (1, 4, cfg.num_embed)
    assert bool(jnp.all(jnp.isfinite(logits)))


@pytest.mark.parametrize("flag", ["use_flash_attention", "use_fused_attention"])
def test_attention_backend_flags_keep_split_rope_nope_forward_valid(flag):
    cfg = dataclasses.replace(make_small_config(), **{flag: True})
    model = Transformer(cfg)

    tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    variables = model.init(jax.random.PRNGKey(0), tokens)
    logits = model.apply(variables, tokens)

    assert logits.shape == (1, 4, cfg.num_embed)
    assert bool(jnp.all(jnp.isfinite(logits)))
