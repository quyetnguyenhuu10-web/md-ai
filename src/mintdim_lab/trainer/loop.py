"""Trainer loop orchestration.

This module orchestrates:
- selecting one explicit JAX runtime adapter by parameter
- reading records from the external public mintdim unit-read API
- converting unit_read records to JAX Batch objects
- placing batches on the selected runtime device
- optionally compiling the update step
- calling the update step and synchronizing metrics

unit_read is provided by the external public ``mintdim`` package. This repo
does not implement unit_read internally.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

from mintdim_lab.corpus.batch import from_unit_read, stack_batches, to_jax_batch
from mintdim_lab.corpus.mintdim_unit_read import unit_read_records
from mintdim_lab.trainer.state import Batch, StepMetrics, TrainState

RuntimeModule = Any
UnitReadRecord = dict[str, Any]


def select_runtime(name: str) -> RuntimeModule:
    """Select one explicit JAX runtime module by name.

    Supported names:
        cpu
        gpu
        tpu

    This function lives in training orchestration. The runtime adapter modules
    themselves do not auto-select or fall back across platforms.
    """
    normalized = str(name).strip().lower()

    if normalized == "cpu":
        from mintdim_lab.system import jax_cpu

        return jax_cpu

    if normalized == "gpu":
        from mintdim_lab.system import jax_gpu

        return jax_gpu

    if normalized == "tpu":
        from mintdim_lab.system import jax_tpu

        return jax_tpu

    raise ValueError(f"Unsupported runtime {name!r}. Expected one of: 'cpu', 'gpu', 'tpu'.")


def require_runtime_device(
    *,
    runtime_name: str,
    device_index: int = 0,
    local: bool = True,
) -> tuple[RuntimeModule, Any]:
    """Select a runtime module and require one explicit device from it."""
    runtime = select_runtime(runtime_name)
    device = runtime.require_device(index=device_index, local=local)
    return runtime, device


def micro_batch_windows(
    records: Iterable[UnitReadRecord],
    *,
    accum: int,
) -> Iterator[list[Batch]]:
    """Group unit_read records into accumulation windows."""
    if accum <= 0:
        raise ValueError("accum must be positive.")

    window: list[Batch] = []

    for record in records:
        batch = to_jax_batch(from_unit_read(record))
        window.append(batch)

        if len(window) == accum:
            yield window
            window = []

    if window:
        yield window


def run_training_loop(
    *,
    state: TrainState,
    records: Iterable[UnitReadRecord],
    update_step: Callable[[TrainState, Iterable[Batch]], tuple[TrainState, StepMetrics]],
    accum: int,
    runtime_name: str = "cpu",
    device_index: int = 0,
    local: bool = True,
    compile_update: bool = True,
    runtime: RuntimeModule | None = None,
    device: Any | None = None,
) -> TrainState:
    """Run a minimal training loop.

    records:
        Iterable of unit_read records. Each record must contain input_ids,
        target_ids, and target_mask.

    update_step:
        Usually returned by mintdim_lab.trainer.update_step.make_update_step(...).

    runtime_name/device_index/local:
        Used when runtime and device are not explicitly supplied.

    compile_update:
        If true, wrap update_step with runtime.compile_callable(...) once before
        the loop.

    runtime/device:
        Optional explicit runtime module and device. These are useful for tests
        or advanced callers. When omitted, runtime_name selects the module and
        require_device selects the device.
    """
    if runtime is None or device is None:
        selected_runtime, selected_device = require_runtime_device(
            runtime_name=runtime_name,
            device_index=device_index,
            local=local,
        )

        if runtime is None:
            runtime = selected_runtime

        if device is None:
            device = selected_device

    step_fn = update_step
    if compile_update:
        step_fn = runtime.compile_callable(step_fn)

    state = runtime.put_tree(state, device)

    for micro_batches in micro_batch_windows(records, accum=accum):
        placed_batches = runtime.put_tree(stack_batches(micro_batches), device)

        state, metrics = step_fn(state, placed_batches)
        metrics = runtime.sync(metrics)

        _ = metrics

    return state


def run_training_loop_from_unit_read(
    *,
    state: TrainState,
    entries: Sequence[dict[str, Any]],
    sequence_template: Sequence[str],
    target_fields: Sequence[str],
    target_index: Sequence[int] = (),
    update_step: Callable[[TrainState, Iterable[Batch]], tuple[TrainState, StepMetrics]],
    accum: int,
    ignore_id: int = -100,
    runtime_name: str = "cpu",
    device_index: int = 0,
    local: bool = True,
    compile_update: bool = True,
) -> TrainState:
    """Run training using external public mintdim unit-read records."""
    records = unit_read_records(
        entries=entries,
        sequence_template=sequence_template,
        target_fields=target_fields,
        target_index=target_index,
        ignore_id=ignore_id,
    )

    return run_training_loop(
        state=state,
        records=records,
        update_step=update_step,
        accum=accum,
        runtime_name=runtime_name,
        device_index=device_index,
        local=local,
        compile_update=compile_update,
    )

__all__ = [
    "RuntimeModule",
    "UnitReadRecord",
    "micro_batch_windows",
    "require_runtime_device",
    "run_training_loop",
    "run_training_loop_from_unit_read",
    "select_runtime",
    "unit_read_records",
]
