from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_benchmark_runner_has_template_section_contract():
    body = text("src/mintdim_lab/evaluator/config_schema.py")

    assert '"template": {' in body
    assert '"sequence": "template_sequence"' in body
    assert '"input_until": "template_input_until"' in body
    assert '"target": "template_target"' in body
    assert '"stop_at_target_length": "stop_at_target_length"' in body
    assert '"input_key"' not in body
    assert '"target_key"' not in body


def test_benchmark_logic_has_no_prompt_answer_prefix_hardcode():
    body = text("src/mintdim_lab/evaluator/template.py")

    assert "prompt_answer_prefix" not in body
    assert "prompt_already_has_answer_prefix" not in body
    assert "encode_prompt_with_separator" not in body
    assert "class BenchmarkTemplate" in body
    assert "def render_template_text(" in body
    assert "def encode_benchmark_prompt(" in body


def test_benchmark_template_can_stop_before_target_placeholder():
    from mintdim_lab.evaluator import template as module

    template = module.build_benchmark_template(
        sequence=["#Prompt_user: {prompt}\\n#Answer:\\n{answer}"],
        input_until={"field": "answer"},
        target={"field": "answer"},
    )
    rendered = module.render_template_text(
        sequence=template.sequence,
        fields={"prompt": "1+1?", "answer": "2"},
        stop_before_field=template.input_until_field,
    )

    assert rendered == "#Prompt_user: 1+1?\\n#Answer:\\n"


def test_benchmark_math_config_uses_template_contract():
    body = text("recipes/evaluation/math_exact.yaml")

    assert "template:" in body
    assert "sequence:" in body
    assert "input_until:" in body
    assert "target:" in body
    assert "stop_at_target_length: true" in body
