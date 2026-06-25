"""Top-level CLI dispatcher for mintdim-lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mintdim_lab.cli.commands import (  # noqa: E402
    build_units,
    chat,
    evaluate,
    params,
    serve,
    train,
    train_tokenizer,
)
from mintdim_lab.cli.program import MODULE_PROG  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        parser = _parser()
        parser.print_help()
        return 0

    command = args[0]
    rest = args[1:]

    if command == "build-units":
        return build_units.main(rest)

    if command == "train":
        return train.main(rest)

    if command == "train-tokenizer":
        return train_tokenizer.main(rest)

    if command == "eval":
        return evaluate.main(rest)

    if command == "params":
        return params.main(rest)

    if command == "chat":
        return chat.main(rest)

    if command == "serve":
        return serve.main(rest)

    parser = _parser()
    parser.error(f"unknown command: {command}")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=MODULE_PROG,
        description="MintDim lab command-line interface.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("build-units", "train", "train-tokenizer", "eval", "params", "chat", "serve"),
        help="Command to run.",
    )
    return parser


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
