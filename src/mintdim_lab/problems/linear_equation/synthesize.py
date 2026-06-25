"""Deterministic synthesis for linear-equation samples."""

from __future__ import annotations

from .spec import LinearEquation


def generate_unique_equations(existing: set[str]) -> list[LinearEquation]:
    """Generate deterministic filler equations not present in `existing`."""
    output: list[LinearEquation] = []
    seen = set(existing)

    for x_value in range(1, 31):
        for a in range(2, 13):
            direct = LinearEquation(a=a, op=None, b=0, c=a * x_value)
            if direct.compact not in seen:
                seen.add(direct.compact)
                output.append(direct)

            for b in range(1, 21):
                plus = LinearEquation(a=a, op="+", b=b, c=(a * x_value) + b)
                if plus.compact not in seen:
                    seen.add(plus.compact)
                    output.append(plus)

                c = (a * x_value) - b
                if c < 0:
                    continue
                minus = LinearEquation(a=a, op="-", b=b, c=c)
                if minus.compact not in seen:
                    seen.add(minus.compact)
                    output.append(minus)

    return output


__all__ = [
    "generate_unique_equations",
]
