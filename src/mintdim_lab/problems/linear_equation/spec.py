"""Linear-equation problem specification."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

EQUATION_RE = re.compile(r"(?P<a>\d+)x\s*(?:(?P<op>[+-])\s*(?P<b>\d+))?\s*=\s*(?P<c>\d+)")


@dataclass(frozen=True)
class LinearEquation:
    """One-variable positive-integer linear equation."""

    a: int
    op: str | None
    b: int
    c: int

    @property
    def rhs(self) -> int:
        if self.op == "+":
            return self.c - self.b
        if self.op == "-":
            return self.c + self.b
        return self.c

    @property
    def compact(self) -> str:
        if self.op is None:
            return f"{self.a}x={self.c}"
        return f"{self.a}x{self.op}{self.b}={self.c}"

    @property
    def pretty(self) -> str:
        if self.op is None:
            return f"{self.a}x = {self.c}"
        return f"{self.a}x {self.op} {self.b} = {self.c}"

    @property
    def isolated_compact(self) -> str:
        return f"{self.a}x={self.rhs}"

    @property
    def isolated_pretty(self) -> str:
        return f"{self.a}x = {self.rhs}"


def parse_equation(text: str) -> LinearEquation:
    """Parse the first supported equation found in text."""
    match = EQUATION_RE.search(str(text))
    if match is None:
        raise ValueError(f"Could not find a linear equation in: {text!r}")
    op = match.group("op")
    b = int(match.group("b")) if match.group("b") is not None else 0
    return LinearEquation(
        a=int(match.group("a")),
        op=op,
        b=b,
        c=int(match.group("c")),
    )


def reduced_solution(eq: LinearEquation) -> str:
    """Return the exact reduced solution value for x."""
    numerator = int(eq.rhs)
    denominator = int(eq.a)
    gcd = math.gcd(abs(numerator), abs(denominator))
    numerator //= gcd
    denominator //= gcd
    if denominator == 1:
        return str(numerator)
    return f"{numerator}/{denominator}"


def division_expr(eq: LinearEquation) -> str:
    """Return the unreduced division expression used in step-by-step rendering."""
    return f"{eq.rhs}/{eq.a}"


def equation_from_record(record: dict[str, Any]) -> LinearEquation:
    """Find an equation in the repo's current JSONL record shape."""
    source_text = " ".join(
        str(record.get(field, "")) for field in ("prompt", "target", "answer", "text")
    )
    return parse_equation(source_text)


__all__ = [
    "EQUATION_RE",
    "LinearEquation",
    "division_expr",
    "equation_from_record",
    "parse_equation",
    "reduced_solution",
]
