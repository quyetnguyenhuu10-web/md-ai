"""Convert raw JSONL into the current linear-equation training JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render import pretrain_record, sft_record
from .spec import LinearEquation, equation_from_record
from .synthesize import generate_unique_equations


def collect_pretrain_equations(*, input_path: Path, pretrain_count: int) -> list[LinearEquation]:
    """Collect unique equations from JSONL and deterministically fill if needed."""
    equations: list[LinearEquation] = []
    seen_equations: set[str] = set()
    line_number = 0

    with input_path.open(encoding="utf-8") as src:
        for raw in src:
            stripped = raw.strip()
            if not stripped:
                continue
            line_number += 1
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at line {line_number}")
            eq = equation_from_record(record)
            if eq.compact in seen_equations:
                continue
            seen_equations.add(eq.compact)
            equations.append(eq)
            if len(equations) >= int(pretrain_count):
                break

    if len(equations) < int(pretrain_count):
        equations.extend(generate_unique_equations(seen_equations))
        equations = equations[: int(pretrain_count)]
    if len(equations) < int(pretrain_count):
        raise ValueError(f"Need {int(pretrain_count)} unique pretrain equations.")

    return equations


def convert_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    pretrain_count: int,
    sft_count: int,
) -> tuple[int, int, int]:
    """Write pretraining records followed by SFT records."""
    pretrain_equations = collect_pretrain_equations(
        input_path=input_path,
        pretrain_count=int(pretrain_count),
    )
    if int(sft_count) > len(pretrain_equations):
        raise ValueError(
            f"sft_count ({int(sft_count)}) cannot exceed pretrain_count ({len(pretrain_equations)})"
        )

    sft_equations = pretrain_equations[: int(sft_count)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as dst:
        for sample_id, eq in enumerate(pretrain_equations, start=1):
            dst.write(json.dumps(pretrain_record(eq, sample_id=sample_id), ensure_ascii=False))
            dst.write("\n")
        for eq in sft_equations:
            dst.write(json.dumps(sft_record(eq), ensure_ascii=False))
            dst.write("\n")

    total = len(pretrain_equations) + len(sft_equations)
    return total, len(pretrain_equations), len(sft_equations)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite linear-equation JSONL format.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrain-count", type=int, default=1000)
    parser.add_argument("--sft-count", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    total, pretrain, sft = convert_jsonl(
        input_path=Path(args.input),
        output_path=Path(args.output),
        pretrain_count=int(args.pretrain_count),
        sft_count=int(args.sft_count),
    )
    print(f"Wrote {total} records to {args.output}")
    print(f"pretrain_records: {pretrain}")
    print(f"sft_records: {sft}")


if __name__ == "__main__":
    main()


__all__ = [
    "collect_pretrain_equations",
    "convert_jsonl",
    "main",
    "parse_args",
]
