from __future__ import annotations

import argparse
from pathlib import Path

from mintdim_lab.cli.commands.params import run_params_command
from mintdim_lab.cli.commands.train import build_parser as build_train_parser
from mintdim_lab.cli.main import _parser

REPO = Path(__file__).resolve().parents[2]
TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def test_top_level_cli_command_choices_contract():
    parser = _parser()
    action = parser._actions[1]

    assert tuple(action.choices) == (
        "build-units",
        "train",
        "train-tokenizer",
        "eval",
        "params",
        "chat",
        "serve",
    )


def test_train_cli_argument_surface_contract():
    parser = build_train_parser()
    defaults = parser.parse_args([])

    assert defaults.runtime is None
    assert defaults.device_index is None
    assert defaults.global_device is None
    assert defaults.compile_update is None

    ns = parser.parse_args(
        [
            "--repo",
            ".",
            "--training-config",
            "recipes/train/cpu/linear_equation_tiny_cpu.yaml",
            "--model-config",
            TEST_MODEL_CONFIG,
            "--tokenizer-config",
            "recipes/tokenizers/byte_bpe_512.yaml",
            "--unit-read-config",
            "recipes/corpus/linear_equation_unit96/unit_read.yaml",
            "--runtime",
            "cpu",
            "--max-steps",
            "5",
            "--seed",
            "7",
            "--no-compile-update",
            "--jsonl",
            "runs/train_metrics.jsonl",
            "--determinism-log",
            "runs/determinism.jsonl",
            "--checkpoint-dir",
            "runs/checkpoints",
            "--no-ui",
        ]
    )

    assert ns.runtime == "cpu"
    assert ns.max_steps == 5
    assert ns.seed == 7
    assert ns.compile_update is False
    assert ns.jsonl == "runs/train_metrics.jsonl"
    assert ns.determinism_log == "runs/determinism.jsonl"
    assert ns.checkpoint_dir == "runs/checkpoints"


def test_params_command_reports_test_tiny_model_contract():
    result = run_params_command(
        argparse.Namespace(
            repo=str(REPO),
            model_config=TEST_MODEL_CONFIG,
            tokenizer_config="recipes/tokenizers/byte_bpe_512.yaml",
        )
    )

    assert result["status"] == "ok"
    assert result["model_config"] == TEST_MODEL_CONFIG
    assert result["tokenizer_config"] == "recipes/tokenizers/byte_bpe_512.yaml"
    assert result["tokenizer"]["vocab_size"] == 512
    assert result["model"]["num_embed"] == 512
    assert result["model"]["num_layers"] == 2
    assert result["model"]["embed_dim"] == 32
    assert result["model"]["dense_hidden_dim"] == 64
    assert result["model"]["ffw_activation"] == "swiglu"
