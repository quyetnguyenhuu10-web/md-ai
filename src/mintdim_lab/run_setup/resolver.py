"""Run setup config path resolution.

This module resolves run-level config paths. It intentionally does not load
models, tokenizers, datasets, or optimizer state; those are assembled by the
training bundle layer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from mintdim_lab.corpus.config_paths import resolve_unit_read_config_path


@dataclasses.dataclass(frozen=True)
class ResolvedTrainingConfigPaths:
    training_config: Path
    tokenizer_config: Path
    model_config: Path
    unit_read_config: Path


def resolve_config_path(repo: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(repo) / path


def config_override_or_training_value(
    *,
    repo: str | Path,
    override: str | Path | None,
    training_value: str,
) -> Path:
    if override is not None:
        return resolve_config_path(repo, override)
    return resolve_config_path(repo, training_value)


def resolve_training_config_paths(
    *,
    repo: str | Path,
    training_config: str | Path,
    training: Any,
    tokenizer_config: str | Path | None = None,
    model_config: str | Path | None = None,
    unit_read_config: str | Path | None = None,
) -> ResolvedTrainingConfigPaths:
    repo_path = Path(repo)
    raw_unit_read_config = config_override_or_training_value(
        repo=repo_path,
        override=unit_read_config,
        training_value=str(training.unit_read_config),
    )

    return ResolvedTrainingConfigPaths(
        training_config=resolve_config_path(repo_path, training_config),
        tokenizer_config=config_override_or_training_value(
            repo=repo_path,
            override=tokenizer_config,
            training_value=str(training.tokenizer_config),
        ),
        model_config=config_override_or_training_value(
            repo=repo_path,
            override=model_config,
            training_value=str(training.model_config),
        ),
        unit_read_config=resolve_unit_read_config_path(raw_unit_read_config),
    )


__all__ = [
    "ResolvedTrainingConfigPaths",
    "config_override_or_training_value",
    "resolve_config_path",
    "resolve_training_config_paths",
]
