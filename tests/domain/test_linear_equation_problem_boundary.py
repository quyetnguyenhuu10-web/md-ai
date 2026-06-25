from __future__ import annotations

import importlib
from pathlib import Path

from mintdim_lab.problems import linear_equation

ROOT = Path(__file__).resolve().parents[2]


def test_linear_equation_problem_boundary_exports_domain_api():
    exported = {
        "LinearEquation",
        "parse_equation",
        "reduced_solution",
        "pretrain_prompt",
        "pretrain_target",
        "sft_prompt",
        "sft_target",
        "convert_jsonl",
        "grade_answer",
    }

    for name in exported:
        assert hasattr(linear_equation, name), name


def test_linear_equation_problem_package_is_production_boundary():
    assert linear_equation.__name__ == "mintdim_lab.problems.linear_equation"
    assert linear_equation.LinearEquation.__module__.startswith(
        "mintdim_lab.problems.linear_equation"
    )
    assert linear_equation.parse_equation.__module__.startswith(
        "mintdim_lab.problems.linear_equation"
    )
    assert linear_equation.convert_jsonl.__module__.startswith(
        "mintdim_lab.problems.linear_equation"
    )


def test_split_modules_are_importable_from_production_namespace():
    modules = [
        "mintdim_lab.problems.linear_equation.spec",
        "mintdim_lab.problems.linear_equation.synthesize",
        "mintdim_lab.problems.linear_equation.render",
        "mintdim_lab.problems.linear_equation.grade",
        "mintdim_lab.problems.linear_equation.convert_jsonl",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name).__name__ == module_name


def test_problems_boundary_is_independent_from_generic_evaluator():
    for path in (ROOT / "src" / "mintdim_lab" / "problems").rglob("*.py"):
        body = path.read_text(encoding="utf-8")

        assert "mintdim_lab.evaluator" not in body
