"""Train-run context assembly.

This module owns run-level choices derived from CLI args and loaded training
configuration. The train CLI remains responsible for executing the update loop.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from mintdim_lab.corpus.config_paths import resolve_unit_read_config_path
from mintdim_lab.run_setup.checkpoint_metadata import (
    build_checkpoint_metadata,
    checkpoint_config_files,
    jsonable_unit_read_kwargs,
    optional_repo_path,
    repo_relative_string,
)
from mintdim_lab.run_setup.run_directory import (
    TrainRunDirectory,
    resolve_repo_root,
    resolve_train_run_directory,
)
from mintdim_lab.system.paths import resolve_path_from_base
from mintdim_lab.trainer.config import RuntimeConfig


@dataclasses.dataclass(frozen=True)
class TrainRunContext:
    repo: Path
    max_steps: int
    seed: int
    directory: TrainRunDirectory
    checkpoint_metadata: dict[str, Any]
    checkpoint_config_files: dict[str, Path]


def resolve_training_bundle_kwargs(ns: argparse.Namespace) -> dict[str, Any]:
    repo = resolve_repo_root(ns.repo)
    return {
        "repo": repo,
        "training_config": resolve_path_from_base(repo, ns.training_config),
        "tokenizer_config": optional_repo_path(repo, ns.tokenizer_config),
        "model_config": optional_repo_path(repo, ns.model_config),
        "unit_read_config": optional_repo_path(repo, ns.unit_read_config),
    }


def resolve_train_runtime_config(*, ns: argparse.Namespace, bundle: Any) -> RuntimeConfig:
    """Resolve effective train runtime from training YAML plus CLI overrides."""
    training_runtime = bundle.training.runtime
    config = RuntimeConfig(
        name=str(training_runtime.name if ns.runtime is None else ns.runtime),
        device_index=training_runtime.device_index
        if ns.device_index is None
        else ns.device_index,
        global_device=training_runtime.global_device
        if ns.global_device is None
        else ns.global_device,
        compile_update=training_runtime.compile_update
        if ns.compile_update is None
        else ns.compile_update,
    )
    config.validate()
    return config


def namespace_with_runtime_config(
    *,
    ns: argparse.Namespace,
    runtime: RuntimeConfig,
) -> argparse.Namespace:
    """Return a namespace carrying the effective runtime values."""
    values = vars(ns).copy()
    values.update(
        {
            "runtime": runtime.name,
            "device_index": runtime.device_index,
            "global_device": runtime.global_device,
            "compile_update": runtime.compile_update,
        }
    )
    return argparse.Namespace(**values)


def build_train_run_context(
    *,
    ns: argparse.Namespace,
    bundle: Any,
    repo: Path | None = None,
) -> TrainRunContext:
    repo_path = resolve_repo_root(ns.repo) if repo is None else Path(repo)

    max_steps = int(bundle.training.max_steps)
    if ns.max_steps is not None:
        max_steps = int(ns.max_steps)
        if max_steps <= 0:
            raise ValueError("--max-steps must be positive")

    seed = int(bundle.training.seed if ns.seed is None else ns.seed)
    if seed < 0:
        raise ValueError("--seed must be >= 0")

    if int(ns.log_every) <= 0:
        raise ValueError("--log-every must be positive")
    if int(ns.progress_width) <= 0:
        raise ValueError("--progress-width must be positive")

    directory = resolve_train_run_directory(
        repo=repo_path,
        training_checkpoint_dir=bundle.training.checkpoint_dir,
        checkpoint_dir_override=ns.checkpoint_dir,
        jsonl=ns.jsonl,
        determinism_log=ns.determinism_log,
    )

    metadata = build_checkpoint_metadata(
        repo=repo_path,
        ns=ns,
        bundle=bundle,
        checkpoint_dir=directory.checkpoint_dir,
        max_steps=max_steps,
    )
    config_files = checkpoint_config_files(
        repo=repo_path,
        ns=ns,
        bundle=bundle,
    )

    return TrainRunContext(
        repo=repo_path,
        max_steps=max_steps,
        seed=seed,
        directory=directory,
        checkpoint_metadata=metadata,
        checkpoint_config_files=config_files,
    )


def build_train_result(
    *,
    ns: argparse.Namespace,
    bundle: Any,
    context: TrainRunContext,
    accum_steps: int,
    final_step: int,
    saved_checkpoints: list[int],
    last_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    repo = context.repo
    unit_read_config_path = resolve_unit_read_config_path(
        optional_repo_path(repo, ns.unit_read_config)
        or resolve_path_from_base(repo, bundle.training.unit_read_config)
    )
    unit_read_override_is_absolute = (
        ns.unit_read_config is not None and Path(ns.unit_read_config).is_absolute()
    )
    unit_read_config_display = (
        str(unit_read_config_path)
        if unit_read_override_is_absolute
        else repo_relative_string(repo, unit_read_config_path)
    )
    unit_read_config_override = (
        None
        if ns.unit_read_config is None
        else (
            str(resolve_unit_read_config_path(resolve_path_from_base(repo, ns.unit_read_config)))
            if unit_read_override_is_absolute
            else repo_relative_string(
                repo,
                resolve_unit_read_config_path(resolve_path_from_base(repo, ns.unit_read_config)),
            )
        )
    )

    return {
        "status": "ok",
        "repo": str(repo),
        "runtime": str(ns.runtime),
        "device_index": int(ns.device_index),
        "local": not bool(ns.global_device),
        "compile_update": bool(ns.compile_update),
        "seed": context.seed,
        "config_paths": {
            "training_config": str(resolve_path_from_base(repo, ns.training_config)),
            "model_config": bundle.training.model_config,
            "tokenizer_config": bundle.training.tokenizer_config,
            "unit_read_config": unit_read_config_display,
            "checkpoint_dir": bundle.training.checkpoint_dir,
            "checkpoint_max_to_keep": bundle.training.checkpoint_max_to_keep,
            "checkpoint_dir_override": None
            if ns.checkpoint_dir is None
            else str(resolve_path_from_base(repo, ns.checkpoint_dir)),
            "model_config_override": None
            if ns.model_config is None
            else str(resolve_path_from_base(repo, ns.model_config)),
            "tokenizer_config_override": None
            if ns.tokenizer_config is None
            else str(resolve_path_from_base(repo, ns.tokenizer_config)),
            "unit_read_config_override": unit_read_config_override,
        },
        "max_steps": context.max_steps,
        "gradient_accumulation_steps": accum_steps,
        "final_step": int(final_step),
        "checkpoint": {
            "dir": str(context.directory.checkpoint_dir),
            "save_every": int(bundle.training.save_every),
            "max_to_keep": int(bundle.training.checkpoint_max_to_keep),
            "saved_steps": saved_checkpoints,
        },
        "determinism_log": None
        if context.directory.determinism_log_path is None
        else str(context.directory.determinism_log_path),
        "unit_read": jsonable_unit_read_kwargs(bundle.unit_read.to_unit_read_kwargs()),
        "last_metrics": last_metrics,
    }


__all__ = [
    "TrainRunContext",
    "build_train_result",
    "build_train_run_context",
    "namespace_with_runtime_config",
    "resolve_train_runtime_config",
    "resolve_training_bundle_kwargs",
]
