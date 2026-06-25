from __future__ import annotations

import jax.numpy as jnp
import optax

from mintdim_lab.trainer.state import Batch, TrainState
from mintdim_lab.trainer.update_step import (
    make_grad_loss_sum_fn,
    make_streaming_update_steps,
    make_update_step,
    update_step,
)


def _model_apply(params, input_ids):
    x = input_ids.astype(jnp.float32)
    positive = params["w"] * x
    negative = -positive
    return jnp.stack([negative, positive], axis=-1)


def _batch(values):
    arr = jnp.asarray(values, dtype=jnp.int32)
    return Batch(
        input_ids=arr,
        target_ids=jnp.ones_like(arr, dtype=jnp.int32),
        target_mask=jnp.ones_like(arr, dtype=jnp.float32),
    )


def _state(optimizer):
    params = {"w": jnp.asarray(0.1, dtype=jnp.float32)}
    return TrainState(
        params=params,
        opt_state=optimizer.init(params),
        step=0,
    )


def test_update_step_public_functions_live_in_trainer_boundary():
    assert callable(make_grad_loss_sum_fn)
    assert callable(make_update_step)
    assert callable(make_streaming_update_steps)
    assert callable(update_step)


def test_make_update_step_applies_one_optimizer_update_after_accum_window():
    optimizer = optax.sgd(learning_rate=0.01)
    state = _state(optimizer)
    step_fn = make_update_step(
        model_apply=_model_apply,
        optimizer=optimizer,
    )

    new_state, metrics = step_fn(
        state,
        [
            _batch([[1, 2], [3, 4]]),
            _batch([[5, 6], [7, 8]]),
        ],
    )

    assert new_state.step == 1
    assert int(metrics.effective_batch_size) == 4
    assert float(metrics.token_count) == 8.0
    assert float(new_state.params["w"]) != float(state.params["w"])


def test_direct_update_step_matches_factory_shape():
    optimizer = optax.sgd(learning_rate=0.01)
    state = _state(optimizer)

    new_state, metrics = update_step(
        state,
        [_batch([[1, 2], [3, 4]])],
        model_apply=_model_apply,
        optimizer=optimizer,
    )

    assert new_state.step == 1
    assert int(metrics.effective_batch_size) == 2
    assert float(metrics.token_count) == 4.0


def test_streaming_update_steps_update_once_after_finalize():
    optimizer = optax.sgd(learning_rate=0.01)
    state = _state(optimizer)
    init_accum, accumulate_micro_batch, apply_accumulated_update = make_streaming_update_steps(
        model_apply=_model_apply,
        optimizer=optimizer,
    )

    accum = init_accum(state.params)
    accum = accumulate_micro_batch(state.params, accum, _batch([[1, 2], [3, 4]]))
    assert state.step == 0

    accum = accumulate_micro_batch(state.params, accum, _batch([[5, 6], [7, 8]]))
    new_state, metrics = apply_accumulated_update(state, accum)

    assert new_state.step == 1
    assert int(metrics.effective_batch_size) == 4
    assert float(metrics.token_count) == 8.0
    assert float(new_state.params["w"]) != float(state.params["w"])
