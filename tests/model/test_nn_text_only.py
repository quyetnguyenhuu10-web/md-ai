from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from mintdim_lab.model import Transformer, TransformerConfig, make_attention_types
from mintdim_lab.model.builder import (
    build_text_config,
    load_text_config_yaml,
    read_text_config_yaml,
    text_metadata_fields,
)

TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def make_small_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_embed=32,
        max_seq_len=8,
        num_layers=2,
        embed_dim=32,
        dense_hidden_dim=64,
        ffw_activation="gelu",
        num_heads=4,
        num_local_kv_heads=2,
        num_global_kv_heads=None,
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


def test_transformer_architecture_is_preserved_and_returns_logits_shape():
    cfg = make_small_config()
    model = Transformer(cfg)

    tokens = jnp.asarray(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ],
        dtype=jnp.int32,
    )

    variables = model.init(jax.random.PRNGKey(0), tokens)
    logits = model.apply(variables, tokens)

    assert logits.shape == (2, 4, 32)


@pytest.mark.parametrize("rope_dim", [0, 10])
def test_transformer_supports_nope_and_full_rope_splits(rope_dim):
    cfg = dataclasses.replace(
        make_small_config(),
        local_rope_dim=rope_dim,
        global_rope_dim=rope_dim,
    )
    cfg.validate()
    model = Transformer(cfg)

    tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    variables = model.init(jax.random.PRNGKey(0), tokens)
    logits = model.apply(variables, tokens)

    assert cfg.rope_dim_for_layer(0) == rope_dim
    assert cfg.nope_dim_for_layer(0) == cfg.qk_dim_for_layer(0) - rope_dim
    assert "rope_gate_logit" not in variables["params"]["layer_0"]["attn"]
    assert logits.shape == (1, 4, cfg.num_embed)
    assert bool(jnp.all(jnp.isfinite(logits)))


def test_config_rejects_rope_dim_larger_than_qk_dim():
    cfg = dataclasses.replace(make_small_config(), local_rope_dim=12)

    with pytest.raises(ValueError, match="local_rope_dim"):
        cfg.validate()


def test_config_rejects_odd_positive_rope_dim():
    cfg = dataclasses.replace(make_small_config(), local_rope_dim=3)

    with pytest.raises(ValueError, match="must be even"):
        cfg.validate()


def test_gradient_checkpointing_flag_wraps_blocks_with_remat_and_keeps_gradients_finite():
    cfg = dataclasses.replace(make_small_config(), use_gradient_checkpointing=True)
    model = Transformer(cfg)
    tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    variables = model.init(jax.random.PRNGKey(0), tokens)

    def loss(params):
        return jnp.mean(model.apply({"params": params}, tokens))

    assert "remat" in str(jax.make_jaxpr(loss)(variables["params"]))

    grads = jax.grad(loss)(variables["params"])
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(grads))


def test_build_text_config_preserves_existing_config_builder_contract():
    values = {
        "model": {
            "max_sequence_length": 8,
            "num_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "activation": "gelu",
            "tie_word_embeddings": True,
        },
        "attention": {
            "num_attention_heads": 4,
            "value_head_dim": 8,
            "layer_types": ["global"],
            "local_sliding": {
                "window_size": 4,
                "num_key_value_heads": 2,
                "qk_head_dim": 10,
                "qk_logits_softcap": None,
                "rope": {"dim": 4, "theta": 10000, "scale": 1.0},
            },
            "global": {
                "num_key_value_heads": None,
                "qk_head_dim": 10,
                "qk_logits_softcap": None,
                "rope": {"dim": 4, "theta": 10000, "scale": 1.0},
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

    cfg = build_text_config(
        values,
        preset_name="test",
        num_embed=32,
    )

    assert cfg.num_embed == 32
    assert cfg.num_layers == 2
    assert cfg.attention_types == make_attention_types(["global"], num_layers=2)


def test_build_text_config_supports_per_branch_qk_logits_softcap():
    values = {
        "model": {
            "max_sequence_length": 8,
            "num_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "activation": "gelu",
            "tie_word_embeddings": True,
        },
        "attention": {
            "num_attention_heads": 4,
            "value_head_dim": 8,
            "layer_types": ["local_sliding", "global"],
            "local_sliding": {
                "window_size": 4,
                "num_key_value_heads": 2,
                "qk_head_dim": 10,
                "qk_logits_softcap": 2.0,
                "rope": {"dim": 4, "theta": 10000, "scale": 1.0},
            },
            "global": {
                "num_key_value_heads": 4,
                "qk_head_dim": 10,
                "qk_logits_softcap": 7.0,
                "rope": {"dim": 4, "theta": 10000, "scale": 1.0},
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

    cfg = build_text_config(values, preset_name="test", num_embed=32)

    assert cfg.local_qk_logits_softcap == 2.0
    assert cfg.global_qk_logits_softcap == 7.0
    assert cfg.qk_logits_softcap_for_layer(0) == 2.0
    assert cfg.qk_logits_softcap_for_layer(1) == 7.0


def test_text_metadata_fields_has_no_registry_dependency():
    fields = text_metadata_fields()

    assert "attention_pattern" in fields
    assert "use_gradient_checkpointing" in fields
    assert "num_embed" not in fields


def test_nn_source_has_no_old_project_or_non_text_side_effect_references():
    root = Path(__file__).resolve().parents[2] / "md" / "nn"

    forbidden = (
        "foundation.registry",
        "multimodal",
        "vision",
        "audio",
        "registry.register",
        "register_builder",
        "register_metadata",
    )

    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "mintdim_lab" in text:
            assert "mintdim_lab.model" in text, f"{path} imports outside model boundary"
        for needle in forbidden:
            assert needle not in text, f"{path} still contains {needle!r}"


def test_mtp_fields_are_not_part_of_text_config():
    fields = set(TransformerConfig.__dataclass_fields__)
    metadata = set(text_metadata_fields())

    assert "dense_layers" not in fields
    assert "dense_layers" not in metadata
    assert "mtp_num_future_tokens" not in fields
    assert "mtp_loss_weight" not in fields
    assert "mtp_num_future_tokens" not in metadata
    assert "mtp_loss_weight" not in metadata
    assert not hasattr(TransformerConfig, "use_mtp")


def test_load_text_config_yaml_from_configs_models():
    path = Path(__file__).resolve().parents[2] / TEST_MODEL_CONFIG

    cfg = load_text_config_yaml(path, num_embed=32000)

    assert cfg.num_embed == 32000
    assert cfg.max_seq_len == 128
    assert cfg.num_layers == 2
    assert cfg.embed_dim == 32
    assert cfg.dense_hidden_dim == 64
    assert cfg.use_gradient_checkpointing is False
    assert cfg.use_fused_attention is False
    assert cfg.use_flash_attention is False
    assert cfg.attention_types == make_attention_types(
        ["local_sliding", "global"],
        num_layers=2,
    )


def test_model_yaml_requires_explicit_gradient_checkpointing():
    path = Path(__file__).resolve().parents[2] / TEST_MODEL_CONFIG
    values = read_text_config_yaml(path)
    values["runtime"].pop("gradient_checkpointing")

    with pytest.raises(ValueError, match="runtime.gradient_checkpointing"):
        build_text_config(values, preset_name="missing-gradient-checkpointing", num_embed=32000)


def test_load_text_config_yaml_tolerates_legacy_zero_mtp_fields(tmp_path):
    path = tmp_path / "legacy_zero_mtp.yaml"
    path.write_text(
        """
model:
  max_sequence_length: 8
  num_layers: 2
  hidden_size: 32
  intermediate_size: 64
  activation: gelu
  tie_word_embeddings: true

attention:
  num_attention_heads: 4
  value_head_dim: 8
  layer_types:
    - global
  local_sliding:
    window_size: 4
    num_key_value_heads: 2
    qk_head_dim: 10
    qk_logits_softcap: null
    rope:
      dim: 4
      theta: 10000
      scale: 1.0
  global:
    num_key_value_heads: null
    qk_head_dim: 10
    qk_logits_softcap: null
    rope:
      dim: 4
      theta: 10000
      scale: 1.0

normalization:
  post_attention: false
  post_ffn: false
  qk_norm:
    enabled: false
    learnable_scale: false

output:
  vocab_logits_softcap: null

runtime:
  gradient_checkpointing: false
  flash_attention: false
  fused_attention: false

mtp_num_future_tokens: 0
mtp_loss_weight: 0.0
""",
        encoding="utf-8",
    )

    cfg = load_text_config_yaml(path, num_embed=32)

    assert cfg.num_embed == 32
    assert "mtp_num_future_tokens" not in TransformerConfig.__dataclass_fields__
    assert "mtp_loss_weight" not in TransformerConfig.__dataclass_fields__


def test_load_text_config_yaml_rejects_nonzero_legacy_mtp_fields(tmp_path):
    path = tmp_path / "bad_mtp.yaml"
    path.write_text(
        """
model:
  max_sequence_length: 8
  num_layers: 2
  hidden_size: 32
  intermediate_size: 64
  activation: gelu
  tie_word_embeddings: true

attention:
  num_attention_heads: 4
  value_head_dim: 8
  layer_types:
    - global
  local_sliding:
    window_size: 4
    num_key_value_heads: 2
    qk_head_dim: 10
    qk_logits_softcap: null
    rope:
      dim: 4
      theta: 10000
      scale: 1.0
  global:
    num_key_value_heads: null
    qk_head_dim: 10
    qk_logits_softcap: null
    rope:
      dim: 4
      theta: 10000
      scale: 1.0

normalization:
  post_attention: false
  post_ffn: false
  qk_norm:
    enabled: false
    learnable_scale: false

output:
  vocab_logits_softcap: null

runtime:
  gradient_checkpointing: false
  flash_attention: false
  fused_attention: false

mtp_num_future_tokens: 1
mtp_loss_weight: 0.5
""",
        encoding="utf-8",
    )

    try:
        load_text_config_yaml(path, num_embed=32)
    except ValueError as exc:
        assert "MTP has been removed" in str(exc)
    else:
        raise AssertionError("nonzero legacy MTP fields should be rejected")


def test_attention_forward_runs_with_split_rope_nope_position():
    cfg = make_small_config()
    model = Transformer(cfg)

    tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    variables = model.init(jax.random.PRNGKey(0), tokens)
    logits = model.apply(variables, tokens)

    assert logits.shape == (1, 4, cfg.num_embed)
    assert bool(jnp.all(jnp.isfinite(logits)))


@pytest.mark.parametrize("flag", ["use_flash_attention", "use_fused_attention"])
def test_attention_backend_flags_keep_forward_valid(flag):
    cfg = dataclasses.replace(make_small_config(), **{flag: True})
    model = Transformer(cfg)

    tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    variables = model.init(jax.random.PRNGKey(0), tokens)
    logits = model.apply(variables, tokens)

    assert logits.shape == (1, 4, cfg.num_embed)
    assert bool(jnp.all(jnp.isfinite(logits)))
