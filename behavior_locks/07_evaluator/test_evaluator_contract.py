from __future__ import annotations

from mintdim_lab.evaluator.data import Example
from mintdim_lab.evaluator.generation import Prediction
from mintdim_lab.evaluator.report import format_first_wrong, summarize_predictions
from mintdim_lab.evaluator.scoring import normalize_for_exact
from mintdim_lab.evaluator.template import (
    build_benchmark_template,
    render_template_text,
    target_text,
)


def test_benchmark_template_stops_before_target_contract():
    template = build_benchmark_template(
        sequence=["{prompt}\n{target}<eos>"],
        input_until={"field": "target"},
        target={"field": "target"},
    )
    example = Example(
        prompt="Giải phương trình: 2x-2=3?",
        answer="x=5/2",
        fields={"prompt": "Giải phương trình: 2x-2=3?", "target": "x=5/2"},
    )

    rendered = render_template_text(
        sequence=template.sequence,
        fields=example.fields,
        stop_before_field=template.input_until_field,
    )

    assert rendered == "Giải phương trình: 2x-2=3?\n"
    assert target_text(example, template) == "x=5/2"
    assert template.required_fields() == ("prompt", "target")


def test_exact_scoring_and_wrong_report_contract():
    flags = {
        "ignore_whitespace": True,
        "normalize_line_endings": True,
        "case_insensitive": True,
    }
    assert normalize_for_exact(" X = 5\\n", flags=flags) == "x=5"

    predictions = [
        Prediction(
            Example(prompt="p1", answer="x=1", fields={"prompt": "p1", "target": "x=1"}),
            prediction="x=1",
            correct=True,
        ),
        Prediction(
            Example(prompt="p2", answer="x=2", fields={"prompt": "p2", "target": "x=2"}),
            prediction="x=3",
            correct=False,
        ),
    ]
    summary = summarize_predictions(predictions, elapsed=2.0, wrong_limit=1, scoring_flags=flags)
    wrong = format_first_wrong(summary["first_wrong"])

    assert summary["correct"] == 1
    assert summary["total"] == 2
    assert summary["accuracy"] == 0.5
    assert wrong == [
        {
            "prompt": "p2",
            "answer": "x=2",
            "predict": "x=3",
            "norm": {"answer": "x=2", "predict": "x=3"},
        }
    ]
