"""Resolve corpus recipe directories to canonical config files."""

from __future__ import annotations

from pathlib import Path

_UNIT_BUILD_FILE = "unit_build.yaml"
_UNIT_READ_FILE = "unit_read.yaml"
_ALLOWED_CORPUS_CONFIG_FILES = frozenset({_UNIT_BUILD_FILE, _UNIT_READ_FILE})


def resolve_unit_build_config_path(path: str | Path) -> Path:
    """Resolve a unit-build config file or corpus directory."""
    return resolve_corpus_config_path(path, filename=_UNIT_BUILD_FILE)


def resolve_unit_read_config_path(path: str | Path) -> Path:
    """Resolve a unit-read config file or corpus directory."""
    return resolve_corpus_config_path(path, filename=_UNIT_READ_FILE)


def resolve_corpus_config_path(path: str | Path, *, filename: str) -> Path:
    """Resolve a corpus config path.

    Explicit files are accepted as-is. Directories must use one of the canonical
    corpus filenames and contain the requested file.
    """
    if filename not in _ALLOWED_CORPUS_CONFIG_FILES:
        allowed = ", ".join(sorted(_ALLOWED_CORPUS_CONFIG_FILES))
        raise ValueError(f"corpus config filename must be one of: {allowed}")

    config_path = Path(path)
    if not config_path.is_dir():
        return config_path

    resolved = config_path / filename
    if not resolved.is_file():
        raise FileNotFoundError(f"corpus directory must contain {filename}: {config_path}")

    return resolved


__all__ = [
    "resolve_corpus_config_path",
    "resolve_unit_build_config_path",
    "resolve_unit_read_config_path",
]
