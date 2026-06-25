"""CLI command for running the HTTP inference server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mintdim_lab.cli.program import command_prog
from mintdim_lab.serving.http_api import run_server


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        run_server(parser.parse_args(list(sys.argv[1:] if argv is None else argv)))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog("serve"),
        description="Run the MintDim HTTP inference server.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true")
    return parser


__all__ = ["build_parser", "main"]
