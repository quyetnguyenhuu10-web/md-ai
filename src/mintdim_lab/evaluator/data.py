from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class Example:
    prompt: str
    answer: str
    fields: dict[str, str]


def load_jsonl_examples(
    path: Path,
    *,
    required_fields: Sequence[str],
    target_field: str,
) -> list[Example]:
    examples: list[Example] = []
    required = tuple(str(field) for field in required_fields)

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must be a JSON object")

            fields: dict[str, str] = {}
            for field in required:
                fields[field] = _required_string_field(
                    record,
                    field,
                    line_number=line_number,
                    label=field,
                )

            answer = fields[target_field]
            prompt = fields.get("prompt", "")
            examples.append(Example(prompt=prompt, answer=answer, fields=fields))

    return examples


def _required_string_field(
    record: Mapping[str, Any],
    field_path: str,
    *,
    line_number: int,
    label: str,
) -> str:
    value = _field_value(
        record,
        field_path,
        line_number=line_number,
        label=label,
    )
    if not isinstance(value, str):
        raise ValueError(f"Line {line_number} {label} field {field_path!r} must be a string")
    return value


def _field_value(
    record: Mapping[str, Any],
    field_path: str,
    *,
    line_number: int,
    label: str,
) -> Any:
    if not field_path:
        raise ValueError(f"{label} field path must not be empty")

    current: Any = record
    for part in str(field_path).split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"Line {line_number} missing {label} field path: {field_path!r}")
        current = current[part]
    return current
