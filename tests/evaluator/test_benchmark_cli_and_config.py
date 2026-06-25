from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_top_level_cli_dispatches_eval_command():
    body = text("src/mintdim_lab/cli/main.py")

    assert 'if command == "eval":' in body
    assert "return evaluate.main(rest)" in body
    assert '"eval"' in body


def test_eval_command_module_owns_cli_parse_and_calls_engine():
    body = text("src/mintdim_lab/cli/commands/evaluate.py")
    engine = text("src/mintdim_lab/evaluator/run_evaluation.py")

    assert "from mintdim_lab.evaluator import run_evaluation" in body
    assert "def build_parser(" in body
    assert "run_evaluation.run_tasks(" in body
    assert "def parse_args(" not in engine
    assert "def main(" not in engine


def test_benchmark_cli_accepts_overrides():
    body = text("src/mintdim_lab/cli/commands/evaluate.py")

    assert '"--checkpoint"' in body
    assert '"--vocab-path"' in body
    assert '"--json-path"' in body
    assert 'overrides["checkpoint"] = ns.checkpoint' in body
    assert 'overrides["vocab_path"] = ns.vocab_path' in body
    assert 'overrides["json_output"] = ns.json_path' in body


def test_benchmark_runner_requires_vocab_and_explicit_checkpoint():
    body = text("src/mintdim_lab/evaluator/run_evaluation.py")

    assert '"vocab_path"' in body
    assert "def _resolve_explicit_checkpoint_path(" in body
    assert "not a checkpoint root directory" in body
    assert 'path.name.startswith("step_")' in body
    assert 'checkpoint_path = _resolve_explicit_checkpoint_path(config["checkpoint"])' in body


def test_benchmark_runner_defaults_json_output_to_checkpoint_dir():
    body = text("src/mintdim_lab/evaluator/run_evaluation.py")
    config = text("recipes/evaluation/math_exact.yaml")

    assert "def _resolve_json_output_path(" in body
    assert 'checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent' in body
    assert 'checkpoint_dir / "benchmark" / f"{Path(config_path).stem}.json"' in body
    assert 'config.get("json_output")' in body
    assert "json_path:" not in config
    assert "runs/benchmark" not in config


def test_benchmark_math_config_names_checkpoint_and_vocab_explicitly():
    body = text("recipes/evaluation/math_exact.yaml")

    assert "checkpoint:" in body
    assert "path:" in body
    assert "vocab:" in body
    assert "path: data_store/tokenizers/byte_bpe_512/tokenizer.json" in body
    assert "template:" in body
    assert "input_until:" in body
    assert "field: target" in body
    assert "target:" in body
