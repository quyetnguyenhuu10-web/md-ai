"""CLI command for running training from config files and unit_read data."""

from __future__ import annotations

import argparse
import json
import sys

from mintdim_lab.cli.program import command_prog
from mintdim_lab.trainer.run_training import run_train_command


def main(argv: list[str] | None = None) -> int:
    """Run the train command."""
    parser = build_parser()

    try:
        ns = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
        result = run_train_command(ns)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog("train"),
        description="Run real training from MintDim lab configs and public unit-read data.",
    )

    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--training-config",
        default="recipes/train/cpu/linear_equation_tiny_cpu.yaml",
        help="Training YAML path relative to repo unless absolute.",
    )
    parser.add_argument(
        "--tokenizer-config",
        default=None,
        help="Optional tokenizer YAML override. Defaults to training YAML tokenizer_config.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Optional model YAML override. Defaults to training YAML model_config.",
    )
    parser.add_argument(
        "--unit-read-config",
        default=None,
        help="Optional unit-read YAML override. Defaults to training YAML unit_read_config.",
    )
    parser.add_argument(
        "--runtime",
        default=None,
        choices=("cpu", "gpu", "tpu"),
        help="Optional override for training YAML runtime.name.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional override for training YAML runtime.device_index.",
    )
    parser.add_argument(
        "--global-device",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for training YAML runtime.global_device.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps for this CLI run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional JAX PRNG seed override. Defaults to training YAML seed.",
    )
    parser.add_argument(
        "--compile-update",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override for training YAML runtime.compile_update.",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Optional path to write one JSON object per completed step.",
    )
    parser.add_argument(
        "--determinism-log",
        default=None,
        help=(
            "Optional JSONL path for deterministic-run debugging. "
            "When set, writes run/init/step hashes for batch, params, and optimizer state."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional checkpoint directory override. Defaults to training YAML checkpoint_dir.",
    )
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show terminal progress UI. Default: enabled only when stderr is a TTY.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Replace the active progress row with a text metric log every N completed steps.",
    )
    parser.add_argument(
        "--progress-width",
        type=int,
        default=28,
        help="Terminal progress bar width. Default: 28.",
    )

    return parser


__all__ = ["build_parser", "main", "run_train_command"]


if __name__ == "__main__":
    raise SystemExit(main())
