"""CLI command for building packed training units."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mintdim_lab.cli.program import command_prog
from mintdim_lab.corpus.mintdim_unit_build import (
    resolve_unit_build_config_path,
    run_unit_build_config_yaml,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()

    try:
        ns = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
        result = run_build_units_command(ns)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog("build-units"),
        description="Build packed training units from JSONL data.",
    )
    parser.add_argument(
        "unit_build_config",
        help="Corpus recipe directory or explicit unit-build YAML file.",
    )
    return parser


def run_build_units_command(ns: argparse.Namespace) -> dict[str, Any]:
    raw_config = Path(ns.unit_build_config)
    unit_build_config = resolve_unit_build_config_path(raw_config)
    result = run_unit_build_config_yaml(unit_build_config)

    return {
        "status": "ok",
        "unit_build_input": str(raw_config),
        "unit_build_config": str(unit_build_config),
        "result": result,
    }


__all__ = ["build_parser", "main", "run_build_units_command"]


if __name__ == "__main__":
    raise SystemExit(main())
