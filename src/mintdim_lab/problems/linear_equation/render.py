"""Render linear-equation records for pretraining and SFT."""

from __future__ import annotations

from .spec import LinearEquation, division_expr, reduced_solution


def pretrain_prompt(*, sample_id: int) -> str:
    """Return the unique condition prompt for one pretraining equation."""
    return f"<eq_{int(sample_id):06d}>\n"


def pretrain_target(eq: LinearEquation) -> str:
    """Render compact algebra steps for pretraining."""
    lines = [
        f"eq: {eq.compact}",
    ]
    if eq.op is not None:
        inverse_op = "-" if eq.op == "+" else "+"
        lines.append(f"step: {eq.a}x={eq.c}{inverse_op}{eq.b}")
        lines.append(f"step: {eq.isolated_compact}")
    lines.append(f"step: x={division_expr(eq)}")
    solution = reduced_solution(eq)
    if solution != division_expr(eq):
        lines.append(f"step: x={solution}")
    lines.append(f"ans: x={solution}")
    return "\n".join(lines)


def sft_prompt(eq: LinearEquation) -> str:
    """Render the current Vietnamese SFT prompt."""
    return f"Giải phương trình: {eq.compact}?"


def sft_target(eq: LinearEquation) -> str:
    """Render the current Vietnamese SFT solution explanation."""
    lines: list[str] = []
    if eq.op is not None:
        inverse_op = "-" if eq.op == "+" else "+"
        moved = str(eq.b) if eq.op == "+" else f"-{eq.b}"
        lines.append(
            "Chuyển "
            f"{moved} sang vế phải và đổi dấu, ta được "
            f"{eq.a}x={eq.c}{inverse_op}{eq.b}."
        )
        lines.append(f"Tính vế phải, ta được {eq.isolated_compact}.")
    lines.append(f"Chia hai vế cho {eq.a}, ta được x={division_expr(eq)}.")
    solution = reduced_solution(eq)
    if solution != division_expr(eq):
        lines.append(f"Rút gọn phân số, ta được x={solution}.")
    lines.append(f"Vậy nghiệm là x={solution}.")
    return "\n".join(lines)


def pretrain_record(eq: LinearEquation, *, sample_id: int) -> dict[str, str]:
    """Return one JSONL-ready pretraining record."""
    return {
        "prompt": pretrain_prompt(sample_id=sample_id),
        "target": pretrain_target(eq),
    }


def sft_record(eq: LinearEquation) -> dict[str, str]:
    """Return one JSONL-ready SFT record."""
    return {
        "prompt": sft_prompt(eq),
        "target": sft_target(eq),
    }


__all__ = [
    "pretrain_prompt",
    "pretrain_record",
    "pretrain_target",
    "sft_prompt",
    "sft_record",
    "sft_target",
]
