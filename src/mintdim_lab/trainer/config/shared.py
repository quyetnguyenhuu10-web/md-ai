from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def read_yaml_mapping(path: str | Path, *, purpose: str) -> dict[str, Any]:
    """Read a YAML file whose top-level object must be a mapping."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(f"PyYAML is required to read {purpose} YAML configs.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        raise ValueError(f"{config_path} is empty.")
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level.")

    return dict(loaded)


def reject_required_only_fields(
    values: Mapping[str, Any],
    *,
    required: frozenset[str],
    label: str,
) -> None:
    """Require exactly the listed fields and reject all others."""
    missing = sorted(name for name in required if name not in values)
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")

    unknown = sorted(name for name in values if name not in required)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")


def expect_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Validate a nested mapping field."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def expect_sequence(value: Any, *, label: str) -> Sequence[Any]:
    """Validate a YAML list-like field.

    Strings are sequences in Python, but they are not valid list fields here.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list.")
    return value


def expect_plain_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def expect_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


__all__ = [
    "expect_bool",
    "expect_mapping",
    "expect_plain_int",
    "expect_sequence",
    "read_yaml_mapping",
    "reject_required_only_fields",
]
