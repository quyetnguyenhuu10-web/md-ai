from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mintdim_lab.system.paths import resolve_repo_path


def load_sectioned_config(
    path: str | Path,
    *,
    section_fields: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    path = resolve_repo_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Benchmark config must be a mapping: {path}")

    unknown_sections = sorted(set(data) - set(section_fields), key=str)
    if unknown_sections:
        joined = ", ".join(str(name) for name in unknown_sections)
        raise ValueError(f"Unknown benchmark config section(s): {joined}")

    result: dict[str, Any] = {}
    for section, fields in section_fields.items():
        raw_section = data.get(section, {})
        if raw_section is None:
            continue
        if not isinstance(raw_section, Mapping):
            raise ValueError(f"Benchmark config section must be a mapping: {section}")

        unknown_keys = sorted(set(raw_section) - set(fields), key=str)
        if unknown_keys:
            joined = ", ".join(str(name) for name in unknown_keys)
            raise ValueError(f"Unknown benchmark config key(s) in {section}: {joined}")

        for yaml_key, config_key in fields.items():
            if yaml_key in raw_section:
                result[config_key] = raw_section[yaml_key]

    return result


def require_config_fields(config: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Benchmark config is missing required field(s): {joined}")


def load_manifest(path: str | Path) -> list[Path]:
    path = resolve_repo_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Manifest must be a mapping: {path}")
    raw_paths = data.get("configs")
    if not isinstance(raw_paths, list):
        raise ValueError(f"Manifest must contain 'configs' list: {path}")
    return [resolve_repo_path(p) for p in raw_paths]
