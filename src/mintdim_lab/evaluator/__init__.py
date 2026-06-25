"""Evaluation and benchmark modules."""

from __future__ import annotations

from .data import Example, load_jsonl_examples
from .generation import Prediction, generate_predictions
from .report import format_first_wrong, print_summary, summarize_predictions
from .run_evaluation import run_task
from .scoring import normalize_for_exact
from .template import (
    BenchmarkTemplate,
    build_benchmark_template,
    encode_benchmark_prompt,
    render_template_text,
    target_text,
)

__all__ = [
    "BenchmarkTemplate",
    "Example",
    "Prediction",
    "build_benchmark_template",
    "encode_benchmark_prompt",
    "format_first_wrong",
    "generate_predictions",
    "load_jsonl_examples",
    "normalize_for_exact",
    "print_summary",
    "render_template_text",
    "run_task",
    "summarize_predictions",
    "target_text",
]
