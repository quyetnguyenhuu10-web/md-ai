"""CLI command for running config-driven evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mintdim_lab.cli.program import command_prog
from mintdim_lab.evaluator import run_evaluation
from mintdim_lab.evaluator.config import load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        ns = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
        config_paths = resolve_config_paths(ns)
        if not config_paths:
            parser.error("Provide at least one --config or --manifest")
        run_evaluation.run_tasks(
            config_paths=config_paths,
            overrides=build_overrides(ns),
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog("eval"),
        description="Run one or more config-driven evaluation tasks.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help="Evaluation YAML config. Can be passed multiple times.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest YAML listing config paths to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override limit for number of eval examples.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint.path. Must be a concrete checkpoint file or step_* dir.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=None,
        help="Override vocab.path.",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="Override output.json_path.",
    )
    return parser


def resolve_config_paths(ns: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if ns.config:
        paths.extend(ns.config)
    if ns.manifest:
        paths.extend(load_manifest(ns.manifest))
    return paths


def build_overrides(ns: argparse.Namespace) -> dict[str, object] | None:
    overrides: dict[str, object] = {}
    if ns.limit is not None:
        overrides["limit"] = ns.limit
    if ns.checkpoint is not None:
        overrides["checkpoint"] = ns.checkpoint
    if ns.vocab_path is not None:
        overrides["vocab_path"] = ns.vocab_path
    if ns.json_path is not None:
        overrides["json_output"] = ns.json_path
    return overrides or None


__all__ = ["build_overrides", "build_parser", "main", "resolve_config_paths"]
