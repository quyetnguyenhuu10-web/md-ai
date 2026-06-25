"""Tokenizer audit helpers."""

from __future__ import annotations

import collections
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TEXT_FIELDS = ("text", "prompt", "answer", "target")


def iter_jsonl_text(input_path: Path, *, text_fields: Iterable[str] = TEXT_FIELDS):
    """Yield string text fields from a JSONL corpus."""
    fields = tuple(text_fields)
    with input_path.open(encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            for field in fields:
                value = record.get(field)
                if isinstance(value, str) and value.strip():
                    yield value


def unicode_codepoints(value: str) -> list[str]:
    """Return Unicode codepoint strings for each character."""
    return [f"U+{ord(char):04X}" for char in value]


def write_unk_report(
    *,
    input_paths: list[Path],
    tokenizer: Any,
    report_path: Path,
    max_items: int,
    log_every: int = 100_000,
) -> tuple[int, int]:
    """Write a JSONL report of characters that encode to `<unk>`."""
    unk_id = tokenizer.token_to_id("<unk>")
    char_counts: collections.Counter[str] = collections.Counter()
    n_lines = 0

    for input_path in input_paths:
        for text in iter_jsonl_text(input_path):
            n_lines += 1
            char_counts.update(text)
            if int(log_every) and n_lines % int(log_every) == 0:
                print(
                    f"unk_audit lines={n_lines:,} unique_chars={len(char_counts):,}",
                    flush=True,
                )

    counts: collections.Counter[str] = collections.Counter()
    for char, count in char_counts.items():
        ids = tokenizer.encode(char).ids
        if unk_id in ids:
            counts[char] = count

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        for char, count in counts.most_common(int(max_items)):
            handle.write(
                json.dumps(
                    {
                        "char": char,
                        "codepoints": unicode_codepoints(char),
                        "count": count,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    return sum(counts.values()), len(counts)


__all__ = [
    "TEXT_FIELDS",
    "iter_jsonl_text",
    "unicode_codepoints",
    "write_unk_report",
]
