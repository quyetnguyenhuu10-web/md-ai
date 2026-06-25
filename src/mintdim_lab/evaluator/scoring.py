from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCORING_FLAG_FIELDS: tuple[str, ...] = (
    "ignore_whitespace",
    "normalize_line_endings",
    "case_insensitive",
)


def normalize_for_exact(text: str, *, flags: Mapping[str, bool]) -> str:
    result = str(text)
    if flags["normalize_line_endings"]:
        result = result.replace("\r\n", "\n").replace("\\n", "\n")
    if flags["case_insensitive"]:
        result = result.lower()
    if flags["ignore_whitespace"]:
        result = "".join(result.split())
    else:
        result = result.rstrip()
    return result


def load_scoring_flags(config: Mapping[str, Any]) -> dict[str, bool]:
    missing = [field for field in SCORING_FLAG_FIELDS if field not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Benchmark scoring config is missing required field(s): {joined}")

    return {field: bool(config[field]) for field in SCORING_FLAG_FIELDS}
