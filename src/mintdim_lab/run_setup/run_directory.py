"""Training run path and output file resolution."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TextIO

from mintdim_lab.run_setup.resolver import resolve_config_path


@dataclasses.dataclass(frozen=True)
class TrainRunDirectory:
    """Resolved filesystem locations for one train CLI run."""

    repo: Path
    checkpoint_dir: Path
    jsonl_path: Path | None
    determinism_log_path: Path | None


def resolve_repo_root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_run_path(repo: Path, value: str | Path) -> Path:
    return resolve_config_path(repo, value)


def optional_run_path(repo: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return resolve_run_path(repo, value)


def resolve_train_run_directory(
    *,
    repo: str | Path,
    training_checkpoint_dir: str | Path,
    checkpoint_dir_override: str | Path | None = None,
    jsonl: str | Path | None = None,
    determinism_log: str | Path | None = None,
) -> TrainRunDirectory:
    repo_path = resolve_repo_root(repo)
    checkpoint_dir = resolve_run_path(
        repo_path,
        checkpoint_dir_override
        if checkpoint_dir_override is not None
        else training_checkpoint_dir,
    )

    return TrainRunDirectory(
        repo=repo_path,
        checkpoint_dir=checkpoint_dir,
        jsonl_path=optional_run_path(repo_path, jsonl),
        determinism_log_path=optional_run_path(repo_path, determinism_log),
    )


def open_text_log(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


__all__ = [
    "TrainRunDirectory",
    "open_text_log",
    "optional_run_path",
    "resolve_repo_root",
    "resolve_run_path",
    "resolve_train_run_directory",
]
