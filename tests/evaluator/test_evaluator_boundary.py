from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EVALUATOR_MODULES = (
    "config",
    "data",
    "generation",
    "report",
    "run_evaluation",
    "scoring",
    "template",
)


def _module_path(module) -> str:
    raw = getattr(module, "__file__", None)
    assert raw is not None
    return Path(raw).resolve().as_posix()


def test_evaluator_modules_resolve_to_production_boundary():
    for name in EVALUATOR_MODULES:
        module = importlib.import_module(f"mintdim_lab.evaluator.{name}")

        assert module.__name__ == f"mintdim_lab.evaluator.{name}"
        assert "/src/mintdim_lab/evaluator/" in _module_path(module)


def test_evaluator_package_is_the_production_namespace():
    package = importlib.import_module("mintdim_lab.evaluator")

    assert package.__name__ == "mintdim_lab.evaluator"
    assert "/src/mintdim_lab/evaluator/" in _module_path(package)


def test_evaluator_public_boundary_imports_core_entrypoints():
    from mintdim_lab.evaluator.config import load_sectioned_config
    from mintdim_lab.evaluator.config_schema import SECTION_FIELDS

    assert callable(load_sectioned_config)
    assert isinstance(SECTION_FIELDS, dict)


def test_evaluator_boundary_is_fully_decoupled_from_problems():
    for path in (ROOT / "src" / "mintdim_lab" / "evaluator").glob("*.py"):
        body = path.read_text(encoding="utf-8")

        assert "mintdim_lab.problems" not in body
        assert "linear_equation" not in body
