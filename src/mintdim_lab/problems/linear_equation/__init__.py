"""Linear-equation problem domain."""

from __future__ import annotations

from .convert_jsonl import collect_pretrain_equations, convert_jsonl
from .grade import extract_solution_value, grade_answer
from .render import (
    pretrain_prompt,
    pretrain_record,
    pretrain_target,
    sft_prompt,
    sft_record,
    sft_target,
)
from .spec import (
    LinearEquation,
    division_expr,
    equation_from_record,
    parse_equation,
    reduced_solution,
)
from .synthesize import generate_unique_equations

__all__ = [
    "LinearEquation",
    "collect_pretrain_equations",
    "convert_jsonl",
    "division_expr",
    "equation_from_record",
    "extract_solution_value",
    "generate_unique_equations",
    "grade_answer",
    "parse_equation",
    "pretrain_prompt",
    "pretrain_record",
    "pretrain_target",
    "reduced_solution",
    "sft_prompt",
    "sft_record",
    "sft_target",
]
