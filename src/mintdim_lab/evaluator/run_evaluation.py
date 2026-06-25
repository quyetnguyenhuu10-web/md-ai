from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from mintdim_lab.evaluator import data, generation, report, template
from mintdim_lab.evaluator.config import (
    load_sectioned_config,
    require_config_fields,
)
from mintdim_lab.evaluator.config_schema import REQUIRED_CONFIG_FIELDS, SECTION_FIELDS
from mintdim_lab.evaluator.scoring import load_scoring_flags
from mintdim_lab.inference.model_loader import (
    config_from_checkpoint_metadata,
    resolve_tokenizer_path_from_metadata,
)
from mintdim_lab.model import Transformer
from mintdim_lab.system.checkpoint_io import (
    checkpoint_logits_head,
    checkpoint_num_embed,
    checkpoint_preset,
    checkpoint_to_variables,
    load_checkpoint,
)
from mintdim_lab.system.paths import REPO_ROOT, resolve_repo_path
from mintdim_lab.tokenizer import Tokenizer, load_tokenizer


def run_tasks(
    *,
    config_paths: list[Path],
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    print(f"Tasks to run: {len(config_paths)}")
    for i, p in enumerate(config_paths, 1):
        print(f"  {i}. {p}")

    results: list[dict[str, Any]] = []
    for i, config_path in enumerate(config_paths, 1):
        print(f"{'=' * 60}")
        print(f"Task {i}/{len(config_paths)}: {config_path}")
        print(f"{'=' * 60}")
        try:
            result = run_task(config_path, overrides=overrides or None)
            results.append(result)
        except SystemExit as exc:
            print(f"FAILED: {exc}", flush=True)
            results.append(
                {
                    "config": str(config_path),
                    "error": str(exc),
                }
            )
        print()

    print_overall_summary(results)
    return results


def checkpoint_step(path: Path) -> int | None:
    match = re.search(r"step_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def print_overall_summary(results: list[dict[str, Any]]) -> None:
    print(f"{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")

    for r in results:
        if "error" in r:
            print(f"  FAIL  {r['config']}: {r['error']}")
        else:
            print(
                f"  {'step=' + str(r.get('step', '?')):<10} "
                f"acc={r['accuracy']:.4f}  "
                f"{r['correct']}/{r['total']}  "
                f"{r['elapsed_sec']:.1f}s  "
                f"{r['config']}"
            )

    ok = [r for r in results if "error" not in r]
    fail = [r for r in results if "error" in r]
    print(f"\n  Total: {len(results)} tasks, {len(ok)} passed, {len(fail)} failed")


__all__ = [
    "REQUIRED_CONFIG_FIELDS",
    "SECTION_FIELDS",
    "checkpoint_step",
    "print_overall_summary",
    "run_task",
    "run_tasks",
]


def _resolve_json_output_path(
    *,
    config_path: str | Path,
    checkpoint_path: Path,
    json_output: str | Path | None,
) -> Path:
    if json_output is not None:
        return resolve_repo_path(json_output)

    checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    return checkpoint_dir / "benchmark" / f"{Path(config_path).stem}.json"


def _step_from_checkpoint(checkpoint: Any, checkpoint_path: Path) -> int | None:
    from mintdim_lab.system.checkpoint_io import checkpoint_metadata

    metadata = checkpoint_metadata(checkpoint)

    value = metadata.get("step")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    return checkpoint_step(checkpoint_path)


def run_task(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_sectioned_config(config_path, section_fields=SECTION_FIELDS)
    if overrides:
        config.update(overrides)
    require_config_fields(config, REQUIRED_CONFIG_FIELDS)

    scoring_flags = load_scoring_flags(config)

    checkpoint_path = _resolve_explicit_checkpoint_path(config["checkpoint"])

    eval_path = resolve_repo_path(config["eval_path"])
    if not eval_path.exists():
        raise SystemExit(f"Eval JSONL not found: {eval_path}")

    prompt_template = template.build_benchmark_template(
        sequence=config["template_sequence"],
        input_until=config["template_input_until"],
        target=config["template_target"],
    )

    examples = data.load_jsonl_examples(
        eval_path,
        required_fields=prompt_template.required_fields(),
        target_field=prompt_template.target_field,
    )
    if int(config["limit"]) > 0:
        examples = examples[: int(config["limit"])]
    if not examples:
        raise SystemExit("No evaluation examples found.")

    started = time.perf_counter()

    checkpoint = load_checkpoint(checkpoint_path)
    preset = checkpoint_preset(checkpoint, default="dense_5m")

    variables = checkpoint_to_variables(checkpoint)
    tokenizer_path = _resolve_tokenizer_path(config.get("vocab_path"), checkpoint)
    tokenizer = load_tokenizer(tokenizer_path)

    _validate_tokenizer_checkpoint(tokenizer, tokenizer_path, checkpoint)

    model_config, metadata_overrides = config_from_checkpoint_metadata(checkpoint)

    expected_num_embed = checkpoint_num_embed(checkpoint, variables=variables)
    if expected_num_embed is not None and expected_num_embed != model_config.num_embed:
        raise SystemExit(
            f"{checkpoint_path}: checkpoint num_embed mismatch: metadata config has "
            f"{model_config.num_embed}, params expect {expected_num_embed}."
        )
    logits_head = checkpoint_logits_head(checkpoint, variables=variables)
    if logits_head != model_config.logits_head:
        raise SystemExit(
            f"{checkpoint_path}: checkpoint logits_head mismatch: metadata config has "
            f"{model_config.logits_head!r}, params imply {logits_head!r}."
        )

    if tokenizer.vocab_size != model_config.num_embed:
        raise SystemExit(
            f"{checkpoint_path}: tokenizer size {tokenizer.vocab_size} != "
            f"config.num_embed {model_config.num_embed}"
        )

    def print_progress(done: int, total: int, correct: int) -> None:
        if correct < 0:
            line = f"Prepared: {done}/{total}"
            print(f"\r{line:<90}", end="", flush=True)
            return
        elapsed_so_far = max(time.perf_counter() - started, 1.0e-9)
        line = (
            f"Measured: {done}/{total} | "
            f"Correct: {correct}/{done} | "
            f"sample/s: {done / elapsed_so_far:.1f}"
        )
        print(f"\r{line:<90}", end="", flush=True)

    model = Transformer(model_config)
    predictions = generation.generate_predictions(
        model=model,
        params=variables["params"],
        config=model_config,
        tokenizer=tokenizer,
        examples=examples,
        prompt_template=prompt_template,
        batch_size=int(config["batch_size"]),
        max_new_tokens=int(config["max_new_tokens"]),
        progress=print_progress,
        fixed_seq_len=config.get("fixed_seq_len"),
        scoring_flags=scoring_flags,
        stop_at_target_length=bool(config.get("stop_at_target_length", False)),
    )
    print(flush=True)

    elapsed = time.perf_counter() - started
    summary = report.summarize_predictions(
        predictions,
        elapsed=elapsed,
        wrong_limit=int(config["wrong_limit"]),
        scoring_flags=scoring_flags,
    )

    report.print_summary(summary)

    seed = _checkpoint_seed_value(checkpoint)
    step = _step_from_checkpoint(checkpoint, checkpoint_path)
    first_wrong = report.format_first_wrong(summary["first_wrong"])

    json_path = _resolve_json_output_path(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        json_output=config.get("json_output"),
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "benchmark": {
                    "checkpoint": _rel(checkpoint_path),
                    "step": step,
                    "seed": seed,
                    "preset": preset,
                    "vocab": _rel(tokenizer_path),
                    "correct": int(summary["correct"]),
                    "total": int(summary["total"]),
                    "wrong_count": int(summary["total"]) - int(summary["correct"]),
                    "accuracy": float(summary["accuracy"]),
                    "elapsed_sec": float(summary["elapsed_sec"]),
                    "examples_per_sec": float(summary["examples_per_sec"]),
                    "fixed_seq_len": config.get("fixed_seq_len"),
                    "stop_at_target_length": bool(config.get("stop_at_target_length", False)),
                    "template": {
                        "sequence": list(prompt_template.sequence),
                        "input_until": {"field": prompt_template.input_until_field},
                        "target": {"field": prompt_template.target_field},
                    },
                    "wrong_limit": int(config["wrong_limit"]),
                    "wrong_count_sampled": summary["wrong_count_sampled"],
                    "first_wrong": first_wrong,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote JSON: {json_path}")

    return {
        "config": _rel(resolve_repo_path(config_path)),
        "checkpoint": _rel(checkpoint_path),
        "json_output": _rel(json_path),
        "vocab": _rel(tokenizer_path),
        "step": step,
        "seed": seed,
        "preset": preset,
        "correct": int(summary["correct"]),
        "total": int(summary["total"]),
        "accuracy": float(summary["accuracy"]),
        "elapsed_sec": float(summary["elapsed_sec"]),
        "examples_per_sec": float(summary["examples_per_sec"]),
        "first_wrong": first_wrong,
    }


def _resolve_explicit_checkpoint_path(value: str | Path) -> Path:
    path = resolve_repo_path(value)
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")

    if path.is_file():
        return path

    if path.is_dir() and path.name.startswith("step_"):
        return path

    if path.is_dir() and (path / "metadata.json").is_file():
        return path

    raise SystemExit(
        "Benchmark checkpoint.path must be a concrete checkpoint file or Orbax step_* "
        f"directory, not a checkpoint root directory: {path}"
    )


def _validate_tokenizer_checkpoint(
    tokenizer: Tokenizer,
    tokenizer_path: Path,
    checkpoint: Any,
) -> None:
    import hashlib

    from mintdim_lab.system.checkpoint_io import checkpoint_metadata

    metadata = checkpoint_metadata(checkpoint)

    raw_size = metadata.get("vocab_size")
    if raw_size is not None and int(raw_size) != int(tokenizer.vocab_size):
        raise SystemExit(
            f"Tokenizer vocab_size {tokenizer.vocab_size} != checkpoint metadata "
            f"vocab_size {raw_size}."
        )

    raw_sha256 = metadata.get("vocab_sha256")
    if isinstance(raw_sha256, str) and raw_sha256.strip():
        actual = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
        if actual != raw_sha256.strip():
            raise SystemExit(
                f"Tokenizer sha256 mismatch for {tokenizer_path}: "
                f"{actual} != checkpoint metadata {raw_sha256}."
            )


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _resolve_tokenizer_path(
    vocab_path: str | Path | None,
    checkpoint: Any,
) -> Path:
    if vocab_path is not None:
        path = resolve_repo_path(vocab_path)
        if path.is_file():
            return path
        raise SystemExit(f"Vocab file not found: {path}")
    return resolve_tokenizer_path_from_metadata(checkpoint)


def _checkpoint_seed_value(checkpoint: Any) -> int | None:
    from mintdim_lab.system.checkpoint_io import checkpoint_metadata

    metadata = checkpoint_metadata(checkpoint)

    for key in ("seed", "init_seed", "rng_seed"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    for section, key in (
        ("training_run", "seed"),
        ("configs", "training"),
    ):
        value = metadata.get(section)
        if not isinstance(value, dict):
            continue
        if key == "training":
            value = value.get(key)
            if not isinstance(value, dict):
                continue
            value = value.get("seed")
        else:
            value = value.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
