from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mintdim_lab.run_setup.bundle import load_training_bundle
from mintdim_lab.run_setup.run_context import (
    build_train_result,
    build_train_run_context,
    namespace_with_runtime_config,
    resolve_train_runtime_config,
    resolve_training_bundle_kwargs,
)

REPO = Path(__file__).resolve().parents[2]
TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def _namespace(**overrides):
    values = {
        "repo": str(REPO),
        "training_config": "recipes/train/cpu/linear_equation_tiny_cpu.yaml",
        "tokenizer_config": None,
        "model_config": TEST_MODEL_CONFIG,
        "unit_read_config": None,
        "runtime": "cpu",
        "device_index": 0,
        "global_device": False,
        "compile_update": False,
        "max_steps": 1,
        "seed": None,
        "jsonl": None,
        "determinism_log": None,
        "checkpoint_dir": None,
        "ui": False,
        "log_every": 100,
        "progress_width": 28,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_context_resolves_training_bundle_kwargs():
    ns = _namespace()
    kwargs = resolve_training_bundle_kwargs(ns)

    assert kwargs["repo"] == REPO
    assert (
        kwargs["training_config"]
        == REPO / "recipes" / "train" / "cpu" / "linear_equation_tiny_cpu.yaml"
    )
    assert kwargs["model_config"] == REPO / TEST_MODEL_CONFIG
    assert kwargs["tokenizer_config"] is None


def test_run_context_resolves_runtime_from_training_yaml_and_cli_overrides():
    ns = _namespace(
        training_config="recipes/train/gpu/linear_equation_tiny_gpu.yaml",
        runtime=None,
        device_index=None,
        global_device=None,
        compile_update=None,
    )
    bundle = load_training_bundle(**resolve_training_bundle_kwargs(ns))

    runtime = resolve_train_runtime_config(ns=ns, bundle=bundle)

    assert runtime.name == "gpu"
    assert runtime.device_index == 0
    assert runtime.global_device is False
    assert runtime.compile_update is True

    override = resolve_train_runtime_config(
        ns=_namespace(
            training_config="recipes/train/gpu/linear_equation_tiny_gpu.yaml",
            runtime="cpu",
            device_index=1,
            global_device=True,
            compile_update=False,
        ),
        bundle=bundle,
    )
    effective_ns = namespace_with_runtime_config(ns=ns, runtime=override)

    assert override.name == "cpu"
    assert override.device_index == 1
    assert override.global_device is True
    assert override.compile_update is False
    assert effective_ns.runtime == "cpu"
    assert effective_ns.device_index == 1


def test_run_context_builds_metadata_and_paths(tmp_path):
    ns = _namespace(
        jsonl=tmp_path / "metrics.jsonl",
        determinism_log=tmp_path / "determinism.jsonl",
    )
    bundle = load_training_bundle(**resolve_training_bundle_kwargs(ns))

    context = build_train_run_context(ns=ns, bundle=bundle, repo=REPO)

    assert context.repo == REPO
    assert context.max_steps == 1
    assert context.seed == 0
    assert context.directory.checkpoint_dir == (
        REPO / "runs" / "train" / "linear_equation_tiny_cpu" / "checkpoints"
    )
    assert context.directory.jsonl_path == tmp_path / "metrics.jsonl"
    assert context.directory.determinism_log_path == tmp_path / "determinism.jsonl"
    assert context.checkpoint_metadata["config_paths"]["training_config"]
    assert context.checkpoint_config_files[
        "recipes/train/cpu/linear_equation_tiny_cpu.yaml"
    ].is_file()


def test_run_context_rejects_invalid_cli_scalars():
    bundle = load_training_bundle(repo=REPO, model_config=TEST_MODEL_CONFIG)

    with pytest.raises(ValueError, match="--max-steps must be positive"):
        build_train_run_context(ns=_namespace(max_steps=0), bundle=bundle, repo=REPO)

    with pytest.raises(ValueError, match="--seed must be >= 0"):
        build_train_run_context(ns=_namespace(seed=-1), bundle=bundle, repo=REPO)

    with pytest.raises(ValueError, match="--log-every must be positive"):
        build_train_run_context(ns=_namespace(log_every=0), bundle=bundle, repo=REPO)

    with pytest.raises(ValueError, match="--progress-width must be positive"):
        build_train_run_context(ns=_namespace(progress_width=0), bundle=bundle, repo=REPO)


def test_run_context_builds_train_result_payload():
    ns = _namespace(seed=7)
    bundle = load_training_bundle(**resolve_training_bundle_kwargs(ns))
    context = build_train_run_context(ns=ns, bundle=bundle, repo=REPO)

    result = build_train_result(
        ns=ns,
        bundle=bundle,
        context=context,
        accum_steps=1,
        final_step=1,
        saved_checkpoints=[1],
        last_metrics={"loss_mean": 1.0},
    )

    assert result["status"] == "ok"
    assert result["repo"] == str(REPO)
    assert result["seed"] == 7
    assert (
        result["config_paths"]["checkpoint_dir"]
        == "runs/train/linear_equation_tiny_cpu/checkpoints"
    )
    assert result["checkpoint"]["saved_steps"] == [1]
    assert result["unit_read"]["target_fields"] == ["target", "token_template"]
    assert result["last_metrics"] == {"loss_mean": 1.0}


def test_run_context_reports_unit_read_directory_override_as_yaml_file():
    ns = _namespace(seed=7, unit_read_config="recipes/corpus/linear_equation_unit96")
    bundle = load_training_bundle(**resolve_training_bundle_kwargs(ns))
    context = build_train_run_context(ns=ns, bundle=bundle, repo=REPO)

    result = build_train_result(
        ns=ns,
        bundle=bundle,
        context=context,
        accum_steps=1,
        final_step=1,
        saved_checkpoints=[],
        last_metrics=None,
    )

    assert (
        result["config_paths"]["unit_read_config"]
        == "recipes/corpus/linear_equation_unit96/unit_read.yaml"
    )
    assert (
        result["config_paths"]["unit_read_config_override"]
        == "recipes/corpus/linear_equation_unit96/unit_read.yaml"
    )
