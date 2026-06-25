"""Train command execution independent from CLI parsing."""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import jax

from mintdim_lab.corpus.batch import from_unit_read, to_jax_batch
from mintdim_lab.run_setup.bundle import init_train_state, load_training_bundle
from mintdim_lab.run_setup.checkpoint_metadata import metadata_with_determinism_debug
from mintdim_lab.run_setup.run_context import (
    build_train_result,
    build_train_run_context,
    namespace_with_runtime_config,
    resolve_train_runtime_config,
    resolve_training_bundle_kwargs,
)
from mintdim_lab.run_setup.run_directory import open_text_log
from mintdim_lab.trainer.checkpointing import (
    save_train_state_checkpoint,
    should_save_checkpoint,
)
from mintdim_lab.trainer.loop import require_runtime_device, unit_read_records
from mintdim_lab.trainer.reproducibility import (
    ordered_hash_sha256,
    runtime_fingerprint,
    state_hashes,
    tree_sha256,
    unit_read_source_fingerprints,
    write_jsonl,
)
from mintdim_lab.trainer.terminal_ui import TerminalTrainUI


def run_train_command(ns: Any) -> dict[str, Any]:
    bundle_kwargs = resolve_training_bundle_kwargs(ns)
    repo = bundle_kwargs["repo"]
    bundle = load_training_bundle(**bundle_kwargs)
    runtime_config = resolve_train_runtime_config(ns=ns, bundle=bundle)
    ns = namespace_with_runtime_config(ns=ns, runtime=runtime_config)
    run_context = build_train_run_context(ns=ns, bundle=bundle, repo=repo)

    max_steps = run_context.max_steps
    seed = run_context.seed

    runtime, device = require_runtime_device(
        runtime_name=str(ns.runtime),
        device_index=int(ns.device_index),
        local=not bool(ns.global_device),
    )

    records = unit_read_records(**bundle.unit_read.to_unit_read_kwargs())
    accum_steps = _unit_read_accum_steps(bundle.unit_read)

    first_record = next(records)
    first_batch = to_jax_batch(from_unit_read(first_record))

    state = init_train_state(
        model=bundle.model,
        optimizer=bundle.optimizer,
        rng=jax.random.PRNGKey(seed),
        input_ids=first_batch.input_ids,
    )
    state = runtime.put_tree(state, device)

    init_accum = bundle.init_accum
    accumulate_micro_batch = bundle.accumulate_micro_batch
    apply_accumulated_update = bundle.apply_accumulated_update
    if bool(ns.compile_update):
        init_accum = runtime.compile_callable(init_accum)
        accumulate_micro_batch = runtime.compile_callable(accumulate_micro_batch)
        apply_accumulated_update = runtime.compile_callable(apply_accumulated_update)

    checkpoint_dir = run_context.directory.checkpoint_dir
    checkpoint_metadata = run_context.checkpoint_metadata
    checkpoint_config_files = run_context.checkpoint_config_files

    ui_enabled = sys.stderr.isatty() if ns.ui is None else bool(ns.ui)
    ui = TerminalTrainUI(
        enabled=ui_enabled,
        total_steps=max_steps,
        log_every=int(ns.log_every),
        bar_width=int(ns.progress_width),
    )

    saved_checkpoints: list[int] = []

    jsonl_handle = None
    if run_context.directory.jsonl_path is not None:
        jsonl_handle = open_text_log(run_context.directory.jsonl_path)

    determinism_handle = None
    determinism_log_path = run_context.directory.determinism_log_path
    if determinism_log_path is not None:
        determinism_handle = open_text_log(determinism_log_path)

    try:
        last_metrics: dict[str, Any] | None = None
        last_determinism_hashes: dict[str, str] | None = None

        if determinism_handle is not None:
            write_jsonl(
                determinism_handle,
                {
                    "event": "run",
                    "seed": seed,
                    "max_steps": int(max_steps),
                    "gradient_accumulation_steps": int(accum_steps),
                    "config_paths": checkpoint_metadata["config_paths"],
                    "config_hashes": checkpoint_metadata["config_hashes"],
                    "runtime": runtime_fingerprint(
                        runtime_name=str(ns.runtime),
                        device=device,
                        compile_update=bool(ns.compile_update),
                        global_device=bool(ns.global_device),
                    ),
                    "unit_read_sources": unit_read_source_fingerprints(
                        entries=bundle.unit_read.entries,
                        repo=repo,
                    ),
                },
            )

            init_hashes = state_hashes(params=state.params, opt_state=state.opt_state)
            write_jsonl(
                determinism_handle,
                {
                    "event": "init",
                    "step": int(state.step),
                    **init_hashes,
                },
            )

        ui.start()
        current_batch = first_batch
        for step_index in range(max_steps):
            accum = init_accum(state.params)
            micro_batch_hashes: list[str] = []
            for accum_index in range(accum_steps):
                micro_batch = (
                    current_batch
                    if accum_index == 0
                    else to_jax_batch(from_unit_read(next(records)))
                )
                if determinism_handle is not None:
                    micro_batch_hashes.append(tree_sha256(micro_batch))
                placed_batch = runtime.put_tree(micro_batch, device)
                accum = accumulate_micro_batch(state.params, accum, placed_batch)

            accum_hash = None
            if determinism_handle is not None:
                accum_hash = tree_sha256(accum)

            state, metrics = apply_accumulated_update(state, accum)
            metrics = runtime.sync(metrics)

            metrics_dict = _metrics_dict(
                step=int(state.step),
                step_index=step_index,
                metrics=metrics,
            )
            last_metrics = metrics_dict
            ui.step(metrics_dict)

            if determinism_handle is not None:
                last_determinism_hashes = state_hashes(
                    params=state.params,
                    opt_state=state.opt_state,
                )
                write_jsonl(
                    determinism_handle,
                    {
                        "event": "step",
                        **metrics_dict,
                        "batch_sha256": ordered_hash_sha256(micro_batch_hashes),
                        "micro_batch_sha256": micro_batch_hashes,
                        "accum_sha256": accum_hash,
                        **last_determinism_hashes,
                    },
                )

            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(metrics_dict, ensure_ascii=False))
                jsonl_handle.write("\n")
                jsonl_handle.flush()

            current_step = int(state.step)
            if should_save_checkpoint(
                step=current_step,
                save_every=bundle.training.save_every,
            ):
                metadata = checkpoint_metadata
                if determinism_handle is not None:
                    metadata = metadata_with_determinism_debug(
                        checkpoint_metadata,
                        log_path=determinism_log_path,
                        repo=repo,
                        step=current_step,
                        hashes=last_determinism_hashes,
                    )
                save_train_state_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    state=state,
                    step=int(state.step),
                    max_to_keep=int(bundle.training.checkpoint_max_to_keep),
                    metadata=metadata,
                    config_files=checkpoint_config_files,
                )
                saved_checkpoints.append(current_step)
                ui.checkpoint(current_step)

            if step_index + 1 < max_steps:
                current_batch = to_jax_batch(from_unit_read(next(records)))

    finally:
        ui.stop()
        if jsonl_handle is not None:
            jsonl_handle.close()
        if determinism_handle is not None:
            determinism_handle.close()

    result = build_train_result(
        ns=ns,
        bundle=bundle,
        context=run_context,
        accum_steps=accum_steps,
        final_step=int(state.step),
        saved_checkpoints=saved_checkpoints,
        last_metrics=last_metrics,
    )

    ui.finish(result)
    return result


def _unit_read_accum_steps(unit_read: Any) -> int:
    values = {int(entry.accum) for entry in unit_read.queue}
    if len(values) != 1:
        raise ValueError(
            "train CLI requires one accum value across unit_read queue entries; "
            f"got {sorted(values)}"
        )
    return values.pop()


def _metrics_dict(*, step: int, step_index: int, metrics: Any) -> dict[str, Any]:
    return {
        "step": step,
        "step_index": int(step_index),
        "effective_batch_size": _finite_int(metrics.effective_batch_size),
        "token_count": _finite_float(metrics.token_count),
        "loss_mean": _finite_float(metrics.loss_mean),
    }


def _finite_int(value: Any) -> int | None:
    out = _finite_float(value)
    if out is None:
        return None
    return int(out)


def _finite_float(value: Any) -> float | None:
    out = float(value)
    if math.isfinite(out):
        return out
    return None


__all__ = ["run_train_command"]
