from __future__ import annotations

import jax.numpy as jnp
import pytest

from mintdim_lab.trainer.loop import (
    micro_batch_windows,
    require_runtime_device,
    run_training_loop,
    select_runtime,
    unit_read_records,
)
from mintdim_lab.trainer.state import Batch, StepMetrics, TrainState


def _record(value: int) -> dict[str, list[list[int]]]:
    return {
        "input_ids": [[value, value + 1]],
        "target_ids": [[value + 1, value + 2]],
        "target_mask": [[1, 1]],
    }


class FakeRuntime:
    def __init__(self):
        self.compile_calls = 0
        self.put_calls = 0
        self.sync_calls = 0
        self.required = []

    def require_device(self, *, index: int = 0, local: bool = True):
        self.required.append((index, local))
        return "fake-device"

    def compile_callable(self, fn):
        self.compile_calls += 1
        return fn

    def put_tree(self, tree, device):
        assert device == "fake-device"
        self.put_calls += 1
        return tree

    def sync(self, value):
        self.sync_calls += 1
        return value


def test_loop_public_functions_live_in_trainer_boundary():
    assert callable(select_runtime)
    assert callable(require_runtime_device)
    assert callable(micro_batch_windows)
    assert callable(run_training_loop)
    assert callable(unit_read_records)


def test_select_runtime_returns_system_runtime_adapters():
    assert select_runtime("cpu").platform() == "cpu"
    assert select_runtime("GPU").platform() == "gpu"
    assert select_runtime(" tpu ").platform() == "tpu"

    with pytest.raises(ValueError, match="Unsupported runtime"):
        select_runtime("banana")


def test_micro_batch_windows_converts_records_to_jax_batches():
    windows = list(
        micro_batch_windows(
            [_record(1), _record(3), _record(5)],
            accum=2,
        )
    )

    assert len(windows) == 2
    assert [len(window) for window in windows] == [2, 1]
    assert all(isinstance(batch, Batch) for window in windows for batch in window)
    assert windows[0][0].input_ids.shape == (1, 2)

    with pytest.raises(ValueError, match="accum must be positive"):
        list(micro_batch_windows([_record(1)], accum=0))


def test_run_training_loop_places_stacked_windows_and_compiles_once():
    runtime = FakeRuntime()
    state = TrainState(params={"w": jnp.asarray(1.0)}, opt_state={}, step=0)
    seen_shapes = []

    def step_fn(state, micro_batches):
        seen_shapes.append(tuple(micro_batches.input_ids.shape))
        return (
            TrainState(
                params=state.params,
                opt_state=state.opt_state,
                step=state.step + 1,
            ),
            StepMetrics(
                loss_mean=jnp.asarray(0.0),
                token_count=jnp.asarray(4.0),
                effective_batch_size=jnp.asarray(2),
            ),
        )

    result = run_training_loop(
        state=state,
        records=[_record(1), _record(3), _record(5), _record(7)],
        update_step=step_fn,
        accum=2,
        runtime=runtime,
        device="fake-device",
        compile_update=True,
    )

    assert result.step == 2
    assert seen_shapes == [(2, 1, 2), (2, 1, 2)]
    assert runtime.compile_calls == 1
    assert runtime.put_calls == 3
    assert runtime.sync_calls == 2
