"""CLI command for training a Byte-BPE tokenizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mintdim_lab.cli.program import command_prog
from mintdim_lab.tokenizer.train_byte_bpe import train_byte_bpe_from_args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        train_byte_bpe_from_args(parser.parse_args(list(sys.argv[1:] if argv is None else argv)))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog("train-tokenizer"),
        description="Build a byte-level BPE vocab from a JSONL corpus.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Path(s) to JSONL with `text`, `target`, or `prompt` and `answer` "
            "string fields."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_store/tokenizers/byte_bpe_512"),
        help="Output directory. Writes <dir>/tokenizer.json.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="Target vocab size including special tokens.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for a pair to be considered for merging.",
    )
    parser.add_argument(
        "--unk-report",
        type=Path,
        default=None,
        help="Optional JSONL report of source characters that encode to <unk>.",
    )
    parser.add_argument(
        "--unk-report-max",
        type=int,
        default=2000,
        help="Maximum number of unique unknown pieces to write to --unk-report.",
    )
    parser.add_argument(
        "--unk-report-log-every",
        type=int,
        default=100_000,
        help="Print unk audit progress every N extracted text lines. Use 0 to disable.",
    )
    return parser


__all__ = ["build_parser", "main"]
