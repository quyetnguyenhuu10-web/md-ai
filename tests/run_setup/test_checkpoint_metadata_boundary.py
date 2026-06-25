from __future__ import annotations

import argparse
from pathlib import Path

from mintdim_lab.run_setup import checkpoint_metadata
from mintdim_lab.run_setup.bundle import load_training_bundle

REPO = Path(__file__).resolve().parents[2]
TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def _namespace() -> argparse.Namespace:
    return argparse.Namespace(
        training_config="recipes/train/cpu/linear_equation_tiny_cpu.yaml",
        tokenizer_config=None,
        model_config=TEST_MODEL_CONFIG,
        unit_read_config=None,
        runtime="cpu",
        device_index=0,
        global_device=False,
        compile_update=False,
        seed=None,
    )


def test_checkpoint_metadata_boundary_builds_rebuild_payload(tmp_path):
    bundle = load_training_bundle(repo=REPO, model_config=TEST_MODEL_CONFIG)
    metadata = checkpoint_metadata.build_checkpoint_metadata(
        repo=REPO,
        ns=_namespace(),
        bundle=bundle,
        checkpoint_dir=tmp_path / "checkpoints",
        max_steps=1,
    )

    assert metadata["metadata_version"] == 1
    assert metadata["format"] == "mintdim_lab.train_state.orbax.v1"
    assert (
        metadata["config_paths"]["training_config"]
        == "recipes/train/cpu/linear_equation_tiny_cpu.yaml"
    )
    assert metadata["config_paths"]["model_config"] == TEST_MODEL_CONFIG
    assert metadata["config_paths"]["checkpoint_dir"].endswith("checkpoints")
    assert metadata["config_hashes"]["training_config"]
    assert metadata["configs"]["training"]["save_every"] == 100
    assert metadata["runtime"]["name"] == "cpu"
    assert metadata["training_run"]["seed"] == 0
    assert metadata["num_embed"] == bundle.tokenizer.vocab_size
    assert metadata["attention_pattern"]


def test_checkpoint_metadata_boundary_maps_config_files_for_sidecars():
    bundle = load_training_bundle(repo=REPO, model_config=TEST_MODEL_CONFIG)
    files = checkpoint_metadata.checkpoint_config_files(
        repo=REPO,
        ns=_namespace(),
        bundle=bundle,
    )

    assert files["recipes/train/cpu/linear_equation_tiny_cpu.yaml"].is_file()
    assert files["recipes/tokenizers/byte_bpe_512.yaml"].is_file()
    assert files[TEST_MODEL_CONFIG].is_file()


def test_checkpoint_metadata_boundary_adds_determinism_debug(tmp_path):
    metadata = {"metadata_version": 1}
    debug = checkpoint_metadata.metadata_with_determinism_debug(
        metadata,
        log_path=tmp_path / "determinism.jsonl",
        repo=REPO,
        step=7,
        hashes={"params_sha256": "a" * 64},
    )

    determinism = debug["debug"]["determinism"]
    assert determinism["enabled"] is True
    assert determinism["step"] == 7
    assert determinism["log_path"].endswith("determinism.jsonl")
    assert determinism["params_sha256"] == "a" * 64


def test_train_runner_imports_metadata_boundary_without_defining_helpers():
    body = (REPO / "src" / "mintdim_lab" / "trainer" / "run_training.py").read_text(
        encoding="utf-8"
    )

    assert "from mintdim_lab.run_setup.checkpoint_metadata import" in body
    assert "metadata_with_determinism_debug(" in body
    assert "from mintdim_lab.run_setup.run_context import" in body
    assert "build_train_run_context(" in body
    assert "build_train_result(" in body
    assert "def build_checkpoint_metadata(" not in body
    assert "def model_config_metadata_values(" not in body


def test_train_cli_delegates_to_runner():
    body = (REPO / "src" / "mintdim_lab" / "cli" / "commands" / "train.py").read_text(
        encoding="utf-8"
    )

    assert "from mintdim_lab.trainer.run_training import run_train_command" in body
    assert "result = run_train_command(ns)" in body
    assert "metadata_with_determinism_debug(" not in body
