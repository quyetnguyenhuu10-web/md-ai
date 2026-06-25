from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_legacy_benchmark_logic_reexport_is_removed() -> None:
    assert not (ROOT / "md" / "benchmark" / "logic.py").exists()


def test_benchmark_pure_modules_do_not_import_runtime_execution() -> None:
    for rel in (
        "src/mintdim_lab/evaluator/data.py",
        "src/mintdim_lab/evaluator/template.py",
        "src/mintdim_lab/evaluator/report.py",
    ):
        source = text(rel)

        assert "import jax" not in source
        assert "make_kv_cache" not in source
        assert "load_checkpoint" not in source
        assert "Transformer(" not in source


def test_benchmark_generation_owns_kv_cache_execution_boundary() -> None:
    source = text("src/mintdim_lab/evaluator/generation.py")

    assert "from mintdim_lab.inference.kv_cache import make_kv_cache" in source
    assert "def generate_predictions(" in source


def test_benchmark_runner_uses_split_boundary_modules() -> None:
    source = text("src/mintdim_lab/evaluator/run_evaluation.py")
    retired_logic_name = "be" + "nch_logic"

    assert retired_logic_name not in source
    assert "template.build_benchmark_template" in source
    assert "data.load_jsonl_examples" in source
    assert "generation.generate_predictions" in source
    assert "report.summarize_predictions" in source
    assert "report.format_first_wrong" in source


def test_benchmark_scoring_has_no_code_defaults() -> None:
    source = text("src/mintdim_lab/evaluator/scoring.py")
    schema = text("src/mintdim_lab/evaluator/config_schema.py")
    config = text("recipes/evaluation/math_exact.yaml")

    assert "DEFAULT_SCORING_FLAGS" not in source
    assert "flags is None" not in source
    assert "SCORING_FLAG_FIELDS" in source

    assert '"ignore_whitespace"' in schema
    assert '"normalize_line_endings"' in schema
    assert '"case_insensitive"' in schema

    assert "ignore_whitespace:" in config
    assert "normalize_line_endings:" in config
    assert "case_insensitive:" in config


def test_benchmark_template_does_not_redeclare_shared_types() -> None:
    source = text("src/mintdim_lab/evaluator/template.py")

    assert "from mintdim_lab.evaluator.data import Example" in source
    assert "class BenchmarkTemplate" in source
    assert "class Example" not in source
    assert "class Prediction" not in source
