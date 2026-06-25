from __future__ import annotations

from pathlib import Path

from mintdim_lab.run_setup.run_directory import (
    open_text_log,
    resolve_repo_root,
    resolve_train_run_directory,
)

REPO = Path(__file__).resolve().parents[2]


def test_run_directory_resolves_repo_relative_paths(tmp_path):
    directory = resolve_train_run_directory(
        repo=tmp_path,
        training_checkpoint_dir="runs/checkpoints",
        checkpoint_dir_override=None,
        jsonl="runs/metrics.jsonl",
        determinism_log="runs/determinism.jsonl",
    )

    assert directory.repo == tmp_path.resolve()
    assert directory.checkpoint_dir == tmp_path / "runs" / "checkpoints"
    assert directory.jsonl_path == tmp_path / "runs" / "metrics.jsonl"
    assert directory.determinism_log_path == tmp_path / "runs" / "determinism.jsonl"


def test_run_directory_respects_absolute_overrides(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    jsonl = tmp_path / "logs" / "metrics.jsonl"

    directory = resolve_train_run_directory(
        repo=REPO,
        training_checkpoint_dir="runs/checkpoints",
        checkpoint_dir_override=checkpoint_dir,
        jsonl=jsonl,
        determinism_log=None,
    )

    assert directory.checkpoint_dir == checkpoint_dir
    assert directory.jsonl_path == jsonl
    assert directory.determinism_log_path is None


def test_open_text_log_creates_parent_directory(tmp_path):
    log_path = tmp_path / "nested" / "metrics.jsonl"

    with open_text_log(log_path) as handle:
        handle.write("{}\n")

    assert log_path.read_text(encoding="utf-8") == "{}\n"


def test_resolve_repo_root_expands_to_absolute_path():
    assert resolve_repo_root(REPO) == REPO.resolve()
