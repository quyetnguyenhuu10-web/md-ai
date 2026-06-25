from __future__ import annotations

from typing import Any

from mintdim_lab.evaluator.generation import Prediction


def prefix_codepoints(text: str, *, limit: int = 12) -> list[int]:
    return [ord(ch) for ch in str(text)[: int(limit)]]


def summarize_predictions(
    predictions: list[Prediction],
    *,
    elapsed: float,
    wrong_limit: int,
    scoring_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    from mintdim_lab.evaluator.scoring import normalize_for_exact

    total = len(predictions)
    correct = sum(1 for item in predictions if item.correct)

    first_wrong: list[dict[str, Any]] = []
    for item in predictions:
        if item.correct or len(first_wrong) >= int(wrong_limit):
            continue

        answer = item.example.answer
        first_wrong.append(
            {
                "prompt": item.example.prompt,
                "prompt_repr": repr(item.example.prompt),
                "prompt_prefix_codepoints": prefix_codepoints(item.example.prompt),
                "answer": answer,
                "answer_repr": repr(answer),
                "answer_norm": normalize_for_exact(answer, flags=scoring_flags),
                "answer_prefix_codepoints": prefix_codepoints(answer),
                "prediction": item.prediction,
                "prediction_repr": repr(item.prediction),
                "prediction_norm": normalize_for_exact(item.prediction, flags=scoring_flags),
                "prediction_prefix_codepoints": prefix_codepoints(item.prediction),
            }
        )

    return {
        "correct": correct,
        "total": total,
        "accuracy": float(correct) / max(1, total),
        "elapsed_sec": elapsed,
        "examples_per_sec": float(total) / max(elapsed, 1.0e-9),
        "first_wrong": first_wrong,
        "wrong_count_sampled": len(first_wrong),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "Generate: "
        f"{summary['correct']}/{summary['total']} "
        f"({100.0 * summary['accuracy']:.2f}%) "
        f"in {summary['elapsed_sec']:.3f}s "
        f"({summary['examples_per_sec']:.1f} examples/s)"
    )

    if summary["first_wrong"]:
        print("\nFirst wrong:")
        for item in summary["first_wrong"]:
            print(
                f"  prompt={item['prompt_repr']} "
                f"expected={item['answer_repr']} "
                f"got={item['prediction_repr']} "
                f"expected_norm={item['answer_norm']!r} "
                f"got_norm={item['prediction_norm']!r} "
                f"ans_ord={item['answer_prefix_codepoints']} "
                f"pred_ord={item['prediction_prefix_codepoints']}",
                flush=True,
            )


def format_first_wrong(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in raw_items:
        result.append(
            {
                "prompt": item["prompt"],
                "answer": item["answer"],
                "predict": item["prediction"],
                "norm": {
                    "answer": item["answer_norm"],
                    "predict": item["prediction_norm"],
                },
            }
        )
    return result
