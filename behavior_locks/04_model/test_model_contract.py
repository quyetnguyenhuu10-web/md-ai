from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from mintdim_lab.model import AttentionType, Transformer, TransformerConfig, build_text_config
from mintdim_lab.model.builder import load_text_config_yaml
from mintdim_lab.model.config import make_attention_types
from mintdim_lab.model.rope import (
    apply_nope_attention_logits,
    apply_rope_to_vector,
    apply_split_rope_nope_attention_logits,
    build_position_qk,
    build_rope_inv_freq,
)

REPO = Path(__file__).resolve().parents[2]
TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def branch_contract_values() -> dict:
    return {
        "model": {
            "max_sequence_length": 128,
            "num_layers": 4,
            "hidden_size": 64,
            "intermediate_size": 128,
            "activation": "swiglu",
            "tie_word_embeddings": True,
        },
        "attention": {
            "num_attention_heads": 4,
            "value_head_dim": 16,
            "layer_types": ["local_sliding", "global"],
            "local_sliding": {
                "window_size": 7,
                "num_key_value_heads": 2,
                "qk_head_dim": 16,
                "qk_logits_softcap": 7.0,
                "rope": {"dim": 6, "theta": 111, "scale": 2.0},
            },
            "global": {
                "num_key_value_heads": 4,
                "qk_head_dim": 20,
                "qk_logits_softcap": 11.0,
                "rope": {"dim": 8, "theta": 222, "scale": 3.0},
            },
        },
        "normalization": {
            "post_attention": False,
            "post_ffn": False,
            "qk_norm": {"enabled": False, "learnable_scale": False},
        },
        "output": {"vocab_logits_softcap": None},
        "runtime": {
            "gradient_checkpointing": False,
            "flash_attention": False,
            "fused_attention": False,
        },
    }


def small_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_embed=32,
        max_seq_len=8,
        num_layers=2,
        embed_dim=32,
        dense_hidden_dim=64,
        ffw_activation="gelu",
        num_heads=4,
        num_local_kv_heads=2,
        num_global_kv_heads=4,
        v_head_dim=8,
        attention_types=make_attention_types(["global"], num_layers=2),
        sliding_window_size=4,
        final_logit_softcap=None,
        local_qk_logits_softcap=None,
        global_qk_logits_softcap=None,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
        qk_norm_enabled=False,
        qk_norm_with_scale=False,
        local_qk_dim=10,
        global_qk_dim=10,
        local_rope_dim=4,
        global_rope_dim=4,
        local_rope_base=10000,
        global_rope_base=10000,
        local_rope_scale=1.0,
        global_rope_scale=1.0,
        logits_head="tied",
        use_gradient_checkpointing=False,
        use_fused_attention=False,
        use_flash_attention=False,
    )
    cfg.validate()
    return cfg


def test_tiny_model_yaml_resolution_contract():
    cfg = load_text_config_yaml(REPO / TEST_MODEL_CONFIG, num_embed=512)

    assert cfg.num_embed == 512
    assert cfg.max_seq_len == 128
    assert cfg.num_layers == 2
    assert cfg.embed_dim == 32
    assert cfg.dense_hidden_dim == 64
    assert [item.value for item in cfg.attention_types] == [
        "local_sliding",
        "global",
    ]
    assert cfg.local_qk_logits_softcap == 7.0
    assert cfg.global_qk_logits_softcap == 11.0
    assert cfg.local_qk_dim == 8
    assert cfg.global_qk_dim == 8
    assert cfg.local_rope_dim == 4
    assert cfg.global_rope_dim == 2
    assert cfg.local_rope_base == 10000
    assert cfg.global_rope_base == 100000
    assert cfg.local_rope_scale == 1.0
    assert cfg.global_rope_scale == 1.0
    assert cfg.qk_norm_enabled is False
    assert cfg.use_gradient_checkpointing is False
    assert cfg.use_flash_attention is False
    assert cfg.use_fused_attention is False


def test_tiny_model_local_global_branch_detail_contract():
    cfg = load_text_config_yaml(REPO / TEST_MODEL_CONFIG, num_embed=512)

    for layer_id in (0,):
        assert cfg.attention_types[layer_id] == AttentionType.LOCAL_SLIDING
        assert cfg.kv_heads_for_layer(layer_id) == 2
        assert cfg.qk_dim_for_layer(layer_id) == 8
        assert cfg.rope_dim_for_layer(layer_id) == 4
        assert cfg.nope_dim_for_layer(layer_id) == 4
        assert cfg.rope_base_for_layer(layer_id) == 10000
        assert cfg.rope_scale_for_layer(layer_id) == 1.0
        assert cfg.qk_logits_softcap_for_layer(layer_id) == 7.0

    for layer_id in (1,):
        assert cfg.attention_types[layer_id] == AttentionType.GLOBAL
        assert cfg.kv_heads_for_layer(layer_id) == 4
        assert cfg.qk_dim_for_layer(layer_id) == 8
        assert cfg.rope_dim_for_layer(layer_id) == 2
        assert cfg.nope_dim_for_layer(layer_id) == 6
        assert cfg.rope_base_for_layer(layer_id) == 100000
        assert cfg.rope_scale_for_layer(layer_id) == 1.0
        assert cfg.qk_logits_softcap_for_layer(layer_id) == 11.0


def test_local_global_branch_values_stay_separate_when_yaml_values_differ_contract():
    cfg = build_text_config(branch_contract_values(), preset_name="branch-contract", num_embed=128)

    assert cfg.attention_types == (
        AttentionType.LOCAL_SLIDING,
        AttentionType.GLOBAL,
        AttentionType.LOCAL_SLIDING,
        AttentionType.GLOBAL,
    )

    for layer_id in (0, 2):
        assert cfg.kv_heads_for_layer(layer_id) == 2
        assert cfg.qk_dim_for_layer(layer_id) == 16
        assert cfg.rope_dim_for_layer(layer_id) == 6
        assert cfg.nope_dim_for_layer(layer_id) == 10
        assert cfg.rope_base_for_layer(layer_id) == 111
        assert cfg.rope_scale_for_layer(layer_id) == 2.0
        assert cfg.qk_logits_softcap_for_layer(layer_id) == 7.0

    for layer_id in (1, 3):
        assert cfg.kv_heads_for_layer(layer_id) == 4
        assert cfg.qk_dim_for_layer(layer_id) == 20
        assert cfg.rope_dim_for_layer(layer_id) == 8
        assert cfg.nope_dim_for_layer(layer_id) == 12
        assert cfg.rope_base_for_layer(layer_id) == 222
        assert cfg.rope_scale_for_layer(layer_id) == 3.0
        assert cfg.qk_logits_softcap_for_layer(layer_id) == 11.0

    bad_softcap = branch_contract_values()
    bad_softcap["attention"]["global"]["qk_logits_softcap"] = 0
    with pytest.raises(ValueError, match="global_qk_logits_softcap"):
        build_text_config(bad_softcap, preset_name="bad-softcap", num_embed=128)

    bad_theta = branch_contract_values()
    bad_theta["attention"]["local_sliding"]["rope"]["theta"] = 0
    with pytest.raises(ValueError, match="local_rope_base"):
        build_text_config(bad_theta, preset_name="bad-theta", num_embed=128)

    bad_scale = branch_contract_values()
    bad_scale["attention"]["global"]["rope"]["scale"] = 0.5
    with pytest.raises(ValueError, match="global_rope_scale"):
        build_text_config(bad_scale, preset_name="bad-scale", num_embed=128)


def test_split_rope_nope_tensor_math_contract():
    q = jnp.asarray(
        [[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], [[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]]],
        dtype=jnp.float32,
    )
    k = jnp.asarray(
        [[[[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]], [[2.5, 3.5, 4.5, 5.5, 6.5, 7.5]]]],
        dtype=jnp.float32,
    )
    positions = jnp.asarray([[0, 3]], dtype=jnp.int32)

    assert jnp.allclose(
        build_rope_inv_freq(head_dim=4, base_frequency=100),
        jnp.asarray([1.0, 0.1], dtype=jnp.float32),
    )

    q_dot, k_dot = build_position_qk(
        q,
        k,
        positions,
        rope_dim=4,
        base_frequency=100,
        scale_factor=2.0,
    )
    expected_q_rope = apply_rope_to_vector(
        q[..., :4],
        positions,
        base_frequency=100,
        scale_factor=2.0,
    )
    expected_k_rope = apply_rope_to_vector(
        k[..., :4],
        positions,
        base_frequency=100,
        scale_factor=2.0,
    )

    assert jnp.allclose(q_dot[..., :4], expected_q_rope)
    assert jnp.allclose(k_dot[..., :4], expected_k_rope)
    assert jnp.allclose(q_dot[..., 4:], q[..., 4:])
    assert jnp.allclose(k_dot[..., 4:], k[..., 4:])

    q_scale_1, _ = build_position_qk(
        q,
        k,
        positions,
        rope_dim=4,
        base_frequency=100,
        scale_factor=1.0,
    )
    q_theta_10000, _ = build_position_qk(
        q,
        k,
        positions,
        rope_dim=4,
        base_frequency=10000,
        scale_factor=2.0,
    )
    assert not bool(jnp.allclose(q_dot[:, 1:, :, :4], q_scale_1[:, 1:, :, :4]))
    assert not bool(jnp.allclose(q_dot[:, 1:, :, :4], q_theta_10000[:, 1:, :, :4]))

    split_logits = apply_split_rope_nope_attention_logits(
        q,
        k,
        positions,
        rope_dim=4,
        base_frequency=100,
        scale_factor=2.0,
    )
    manual_logits = apply_nope_attention_logits(q_dot, k_dot)

    assert jnp.allclose(split_logits, manual_logits, atol=1.0e-5)


def test_transformer_forward_shape_and_rope_validation_contract():
    cfg = small_config()
    model = Transformer(cfg)
    tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    variables = model.init(jax.random.PRNGKey(0), tokens)
    logits = model.apply(variables, tokens)

    assert logits.shape == (1, 4, cfg.num_embed)
    assert bool(jnp.all(jnp.isfinite(logits)))

    with pytest.raises(ValueError, match="local_rope_dim"):
        dataclasses.replace(cfg, local_rope_dim=12).validate()

    with pytest.raises(ValueError, match="must be even"):
        dataclasses.replace(cfg, local_rope_dim=3).validate()
