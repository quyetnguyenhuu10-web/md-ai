"""Checkpoint metadata assembly for prepared training runs."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from mintdim_lab.corpus.config_paths import resolve_unit_read_config_path
from mintdim_lab.model.builder import text_metadata_fields
from mintdim_lab.system.paths import file_sha256, resolve_path_from_base


def build_checkpoint_metadata(
    *,
    repo: Path,
    ns: argparse.Namespace,
    bundle: Any,
    checkpoint_dir: Path,
    max_steps: int,
) -> dict[str, Any]:
    paths = effective_config_paths(repo=repo, ns=ns, bundle=bundle)
    tokenizer_yaml = read_yaml_config(paths["tokenizer_config"])
    model_yaml = read_yaml_config(paths["model_config"])
    unit_read_yaml = read_yaml_config(paths["unit_read_config"])
    training_yaml = read_yaml_config(paths["training_config"])

    model_config = getattr(bundle.model, "config", None)
    model_values = model_config_metadata_values(model_config)

    tokenizer_path = str(getattr(bundle.tokenizer, "path", ""))
    tokenizer_vocab_size = getattr(bundle.tokenizer, "vocab_size", None)

    metadata: dict[str, Any] = {
        "metadata_version": 1,
        "format": "mintdim_lab.train_state.orbax.v1",
        "config_paths": {
            "training_config": repo_relative_string(repo, paths["training_config"]),
            "tokenizer_config": repo_relative_string(repo, paths["tokenizer_config"]),
            "unit_read_config": repo_relative_string(repo, paths["unit_read_config"]),
            "model_config": repo_relative_string(repo, paths["model_config"]),
            "checkpoint_dir": repo_relative_string(repo, checkpoint_dir),
        },
        "config_hashes": {name: file_sha256(path) for name, path in paths.items()},
        "configs": {
            "training": training_yaml,
            "tokenizer": tokenizer_yaml,
            "unit_read": unit_read_yaml,
            "model": model_yaml,
        },
        "unit_read": jsonable(bundle.unit_read.to_unit_read_kwargs()),
        "runtime": {
            "name": ns.runtime,
            "device_index": int(ns.device_index),
            "global_device": bool(ns.global_device),
            "compile_update": bool(ns.compile_update),
        },
        "training_run": {
            "max_steps": int(max_steps),
            "save_every": int(bundle.training.save_every),
            "checkpoint_max_to_keep": int(bundle.training.checkpoint_max_to_keep),
            "seed": effective_seed(ns=ns, bundle=bundle),
        },
        "vocab_path": tokenizer_path,
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": getattr(bundle.tokenizer, "type", None),
        "num_embed": tokenizer_vocab_size,
        "vocab_size": tokenizer_vocab_size,
        "pad_id": getattr(bundle.tokenizer, "pad_id", None),
        "unk_id": getattr(bundle.tokenizer, "unk_id", None),
        "eos_id": getattr(bundle.tokenizer, "eos_id", None),
        "model_config_metadata_fields": list(text_metadata_fields(include_num_embed=True)),
    }

    metadata.update(model_values)
    return jsonable(metadata)


def checkpoint_config_files(
    *,
    repo: Path,
    ns: argparse.Namespace,
    bundle: Any,
) -> dict[str, Path]:
    paths = effective_config_paths(repo=repo, ns=ns, bundle=bundle)
    return {checkpoint_config_destination(repo, path): path for path in paths.values()}


def checkpoint_config_destination(repo: Path, path: Path) -> str:
    resolved_repo = repo.resolve()
    resolved_path = path.resolve()

    try:
        return resolved_path.relative_to(resolved_repo).as_posix()
    except ValueError:
        digest = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()[:12]
        return f"external_configs/{digest}_{resolved_path.name}"


def metadata_with_determinism_debug(
    metadata: dict[str, Any],
    *,
    log_path: Path | None,
    repo: Path,
    step: int,
    hashes: Mapping[str, str] | None,
) -> dict[str, Any]:
    out = dict(metadata)
    out["debug"] = {
        **dict(out.get("debug", {})),
        "determinism": {
            "enabled": True,
            "log_path": None if log_path is None else repo_relative_string(repo, log_path),
            "step": int(step),
            **dict(hashes or {}),
        },
    }
    return jsonable(out)


def effective_seed(*, ns: argparse.Namespace, bundle: Any) -> int:
    return int(bundle.training.seed if ns.seed is None else ns.seed)


def effective_config_paths(
    *,
    repo: Path,
    ns: argparse.Namespace,
    bundle: Any,
) -> dict[str, Path]:
    raw_unit_read_config = (
        optional_repo_path(repo, ns.unit_read_config)
        or resolve_path_from_base(repo, bundle.training.unit_read_config)
    )

    return {
        "training_config": resolve_path_from_base(repo, ns.training_config),
        "tokenizer_config": (
            optional_repo_path(repo, ns.tokenizer_config)
            or resolve_path_from_base(repo, bundle.training.tokenizer_config)
        ),
        "model_config": (
            optional_repo_path(repo, ns.model_config)
            or resolve_path_from_base(repo, bundle.training.model_config)
        ),
        "unit_read_config": resolve_unit_read_config_path(raw_unit_read_config),
    }


def read_yaml_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"checkpoint config metadata source must be a mapping: {path}")
    return jsonable(dict(data))


def model_config_metadata_values(config: Any) -> dict[str, Any]:
    if config is None:
        return {}

    if is_dataclass(config):
        raw = asdict(config)
    else:
        raw = {
            field: getattr(config, field)
            for field in text_metadata_fields(include_num_embed=True)
            if hasattr(config, field)
        }

    if "attention_pattern" not in raw and hasattr(config, "attention_types"):
        raw["attention_pattern"] = [
            getattr(item, "value", str(item)) for item in getattr(config, "attention_types")
        ]

    values = {
        field: raw[field] for field in text_metadata_fields(include_num_embed=True) if field in raw
    }
    return jsonable(values)


def repo_relative_string(repo: Path, path: Path) -> str:
    resolved_repo = repo.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_repo).as_posix()
    except ValueError:
        return str(resolved_path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def optional_repo_path(repo: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return resolve_path_from_base(repo, value)


def jsonable_unit_read_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "entries": values["entries"],
        "sequence_template": list(values["sequence_template"]),
        "target_fields": list(values["target_fields"]),
        "target_index": list(values["target_index"]),
        "ignore_id": int(values["ignore_id"]),
    }


__all__ = [
    "build_checkpoint_metadata",
    "checkpoint_config_destination",
    "checkpoint_config_files",
    "effective_config_paths",
    "effective_seed",
    "jsonable",
    "jsonable_unit_read_kwargs",
    "metadata_with_determinism_debug",
    "model_config_metadata_values",
    "optional_repo_path",
    "read_yaml_config",
    "repo_relative_string",
]
