"""Answer grading helpers for linear equations."""

from __future__ import annotations

import re

from .spec import LinearEquation, reduced_solution


def extract_solution_value(text: str) -> str | None:
    """Extract the last `x=value` style answer from text."""
    matches = re.findall(r"x\s*=\s*(-?\d+(?:/\d+)?)", str(text))
    if not matches:
        return None
    return matches[-1].replace(" ", "")


def grade_answer(eq: LinearEquation, answer: str) -> bool:
    """Return whether answer contains the exact reduced solution."""
    return extract_solution_value(answer) == reduced_solution(eq)


__all__ = [
    "extract_solution_value",
    "grade_answer",
]
