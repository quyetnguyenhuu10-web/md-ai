from __future__ import annotations

from pathlib import Path

import pytest

from mintdim_lab.trainer.config import (
    build_training_config,
    build_unit_read_config,
    load_training_config_yaml,
    load_unit_read_config_yaml,
    validate_units_within_max_seq_len,
)

TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def test_load_training_basic_yaml():
    path = (
        Path(__file__).resolve().parents[2]
        / "recipes"
        / "train"
        / "cpu"
        / "linear_equation_tiny_cpu.yaml"
    )

    cfg = load_training_config_yaml(path)

    assert cfg.model_config.endswith(".yaml")
    assert cfg.tokenizer_config == "recipes/tokenizers/byte_bpe_512.yaml"
    assert cfg.unit_read_config == "recipes/corpus/linear_equation_unit96/unit_read.yaml"
    assert cfg.checkpoint_dir == "runs/train/linear_equation_tiny_cpu/checkpoints"
    assert cfg.checkpoint_max_to_keep == 15
    assert cfg.max_steps == 600
    assert cfg.seed == 0
    assert cfg.log_every == 10
    assert cfg.save_every == 100
    assert cfg.runtime.name == "cpu"
    assert cfg.runtime.device_index == 0
    assert cfg.runtime.global_device is False
    assert cfg.runtime.compile_update is True
    assert cfg.optimization.learning_rate == 0.003
    assert cfg.optimization.warmup_steps == 90
    assert cfg.optimization.decay_start_step == 600
    assert cfg.optimization.min_lr_ratio == 0.1


def test_training_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown fields"):
        build_training_config(
            {
                "model_config": TEST_MODEL_CONFIG,
                "tokenizer_config": "recipes/tokenizers/byte_bpe_512.yaml",
                "unit_read_config": "recipes/corpus/linear_equation_unit96/unit_read.yaml",
                "checkpoint_dir": "runs/train/linear_equation_tiny_cpu/checkpoints",
                "checkpoint_max_to_keep": 3,
                "max_steps": 1,
                "seed": 0,
                "log_every": 1,
                "save_every": 1,
                "extra": "no",
                "runtime": {
                    "name": "cpu",
                    "device_index": 0,
                    "global_device": False,
                    "compile_update": True,
                },
                "optimization": {
                    "learning_rate": 1e-3,
                    "warmup_steps": 0,
                    "decay_start_step": 1,
                    "weight_decay": 0.0,
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "adam_eps": 1e-8,
                    "clip_grad_norm": 1.0,
                    "min_lr_ratio": 0.1,
                },
            }
        )


def test_load_unit_read_yaml_queue_list():
    path = (
        Path(__file__).resolve().parents[2]
        / "recipes"
        / "corpus"
        / "linear_equation_unit96"
        / "unit_read.yaml"
    )

    cfg = load_unit_read_config_yaml(path)

    assert len(cfg.queue) == 1
    assert cfg.queue[0].path == "./data_store/packed/linear_equation_unit96/unit_96"
    assert cfg.queue[0].unit == 96
    assert cfg.queue[0].batch == 50
    assert cfg.queue[0].accum == 30

    assert cfg.sequence_template == (
        "prompt",
        "token_template",
        "target",
        "token_template",
    )
    assert cfg.target_fields == ("target", "token_template")
    assert cfg.target_index == (2, 3)
    assert cfg.ignore_id == -100

    kwargs = cfg.to_unit_read_kwargs()
    assert kwargs == {
        "entries": [
            {
                "path": "./data_store/packed/linear_equation_unit96/unit_96",
                "unit": 96,
                "batch": 50,
                "accum": 30,
            },
        ],
        "sequence_template": (
            "prompt",
            "token_template",
            "target",
            "token_template",
        ),
        "target_fields": ("target", "token_template"),
        "target_index": (2, 3),
        "ignore_id": -100,
    }


def test_load_unit_read_yaml_accepts_corpus_directory():
    path = (
        Path(__file__).resolve().parents[2]
        / "recipes"
        / "corpus"
        / "linear_equation_unit96"
    )

    cfg = load_unit_read_config_yaml(path)

    assert cfg.queue[0].path == "./data_store/packed/linear_equation_unit96/unit_96"
    assert cfg.queue[0].unit == 96


def test_unit_read_config_accepts_multi_queue():
    cfg = build_unit_read_config(
        {
            "queue": [
                {
                    "path": "./data_store/packed/math/unit_16",
                    "unit": 16,
                    "batch": 2,
                    "accum": 1,
                },
                {
                    "path": "./data_store/packed/math/unit_32",
                    "unit": 32,
                    "batch": 1,
                    "accum": 2,
                },
            ],
            "layout": {
                "sequence_template": [
                    "token_template",
                    "prompt",
                    "token_template",
                    "answer",
                ],
            },
            "target": {
                "fields": ["answer"],
                "index": [3],
                "ignore_id": -100,
            },
        }
    )

    assert [entry.unit for entry in cfg.queue] == [16, 32]
    assert cfg.target_fields == ("answer",)
    assert cfg.target_index == (3,)


def test_unit_read_config_rejects_mismatched_field_index_pair():
    with pytest.raises(ValueError, match="field/index pairs"):
        build_unit_read_config(
            {
                "queue": [
                    {
                        "path": "./data_store/packed/math/unit_32",
                        "unit": 32,
                        "batch": 1,
                        "accum": 1,
                    }
                ],
                "layout": {
                    "sequence_template": [
                        "token_template",
                        "prompt",
                        "token_template",
                        "answer",
                        "token_template",
                    ],
                },
                "target": {
                    "fields": ["answer"],
                    "index": [4],
                    "ignore_id": -100,
                },
            }
        )


def test_unit_read_config_rejects_non_list_queue():
    with pytest.raises(ValueError, match="queue must be a list"):
        build_unit_read_config(
            {
                "queue": {
                    "path": "./data_store/packed/math/unit_16",
                    "unit": 16,
                    "batch": 2,
                    "accum": 1,
                },
                "layout": {
                    "sequence_template": [
                        "token_template",
                        "prompt",
                        "token_template",
                        "answer",
                    ],
                },
                "target": {
                    "fields": ["answer"],
                    "index": [3],
                    "ignore_id": -100,
                },
            }
        )


def test_unit_read_config_rejects_one_base_index_for_answer():
    with pytest.raises(ValueError, match="0-base"):
        build_unit_read_config(
            {
                "queue": [
                    {
                        "path": "./data_store/packed/math/unit_16",
                        "unit": 16,
                        "batch": 2,
                        "accum": 1,
                    }
                ],
                "layout": {
                    "sequence_template": [
                        "token_template",
                        "prompt",
                        "token_template",
                        "answer",
                    ],
                },
                "target": {
                    "fields": ["answer"],
                    "index": [4],
                    "ignore_id": -100,
                },
            }
        )


def test_unit_read_units_validate_against_model_context_ceiling():
    cfg = build_unit_read_config(
        {
            "queue": [
                {
                    "path": "./data_store/packed/math/unit_16",
                    "unit": 16,
                    "batch": 2,
                    "accum": 1,
                },
                {
                    "path": "./data_store/packed/math/unit_32",
                    "unit": 32,
                    "batch": 1,
                    "accum": 2,
                },
            ],
            "layout": {
                "sequence_template": [
                    "token_template",
                    "prompt",
                    "token_template",
                    "answer",
                ],
            },
            "target": {
                "fields": ["answer"],
                "index": [3],
                "ignore_id": -100,
            },
        }
    )

    validate_units_within_max_seq_len(cfg, max_seq_len=32)


def test_unit_read_units_reject_unit_larger_than_model_context_ceiling():
    cfg = build_unit_read_config(
        {
            "queue": [
                {
                    "path": "./data_store/packed/math/unit_32",
                    "unit": 32,
                    "batch": 1,
                    "accum": 2,
                },
            ],
            "layout": {
                "sequence_template": [
                    "token_template",
                    "prompt",
                    "token_template",
                    "answer",
                ],
            },
            "target": {
                "fields": ["answer"],
                "index": [3],
                "ignore_id": -100,
            },
        }
    )

    with pytest.raises(ValueError, match="must be <= model max_seq_len"):
        validate_units_within_max_seq_len(cfg, max_seq_len=16)
