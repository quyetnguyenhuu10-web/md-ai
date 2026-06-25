"""Adapter for the public MintDim unit-build pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mintdim_lab.corpus.config_paths import resolve_unit_build_config_path


def unit_build_pipeline() -> Any:
    """Return the public `mintdim` unit-build pipeline."""
    from mintdim import pipeline

    return pipeline("unit-build")


def load_unit_build_config_yaml(path: str | Path) -> dict[str, Any]:
    """Load a corpus unit-build YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("PyYAML is required to read unit_build.yaml.") from exc

    config_path = resolve_unit_build_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")

    return loaded


def run_unit_build_config(config: dict[str, Any]) -> Any:
    """Run the public `mintdim` unit-build API from a YAML config mapping."""
    source = config["source"]["jsonl"]
    tokenizer = config["tokenizer"]["hf_json"]
    units = config["units"]
    output = config["output"]["dir"]

    return (
        unit_build_pipeline()
        .source.jsonl(
            files=list(source["files"]),
            fields=[list(group) for group in source["fields"]],
            templates=list(source["templates"]),
        )
        .tokenizer.hf_json(
            files=list(tokenizer["files"]),
        )
        .units(
            sizes=[list(group) for group in units["sizes"]],
            build_batch=list(units["build_batch"]),
        )
        .output.dir(
            output["path"],
            samples_per_shard=list(output["samples_per_shard"]),
        )
        .run(**dict(config.get("run") or {}))
    )


def run_unit_build_config_yaml(path: str | Path) -> Any:
    """Load and run a corpus unit-build YAML file."""
    return run_unit_build_config(load_unit_build_config_yaml(path))


__all__ = [
    "load_unit_build_config_yaml",
    "run_unit_build_config",
    "run_unit_build_config_yaml",
    "resolve_unit_build_config_path",
    "unit_build_pipeline",
]
