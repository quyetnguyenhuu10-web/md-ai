"""Trainer update step composition.

This module owns the composition of loss, gradients, accumulation, and
optimizer update. Runtime/device placement and compilation are handled by the
caller through mintdim_lab.system runtime adapters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import jax
import optax

from mintdim_lab.trainer.gradient_accumulation import (
    accumulate_one_token_normalized_grad,
    accumulate_stacked_token_normalized_grads,
    accumulate_token_normalized_grads,
    finalize_token_normalized_grads,
    init_grad_accum_state,
)
from mintdim_lab.trainer.objective import loss_sum_and_stats
from mintdim_lab.trainer.state import Batch, GradAccumState, StepMetrics, TrainState


def make_grad_loss_sum_fn(
    *,
    model_apply: Callable[[Any, Any], Any],
):
    """Build a function returning grad(loss_sum) for one micro-batch."""

    def wrapped(params, batch: Batch):
        return loss_sum_and_stats(
            params,
            batch,
            model_apply=model_apply,
        )

    return jax.value_and_grad(wrapped, has_aux=True)


def make_update_step(
    *,
    model_apply: Callable[[Any, Any], Any],
    optimizer: optax.GradientTransformation,
):
    """Build one optimizer update function.

    model_apply and optimizer are captured in the closure so the returned
    function has a cleaner shape for runtime.compile_callable():

        update_step(state, micro_batches) -> (state, metrics)
    """
    grad_loss_sum_fn = make_grad_loss_sum_fn(model_apply=model_apply)

    def update_step_inner(
        state: TrainState,
        micro_batches: Iterable[Batch],
    ) -> tuple[TrainState, StepMetrics]:
        if isinstance(micro_batches, Batch):
            accum = accumulate_stacked_token_normalized_grads(
                params=state.params,
                micro_batches=micro_batches,
                grad_loss_sum_fn=grad_loss_sum_fn,
            )
        else:
            accum = accumulate_token_normalized_grads(
                params=state.params,
                micro_batches=micro_batches,
                grad_loss_sum_fn=grad_loss_sum_fn,
            )

        updates, opt_state = optimizer.update(
            accum.grads,
            state.opt_state,
            state.params,
        )
        params = optax.apply_updates(state.params, updates)

        new_state = TrainState(
            params=params,
            opt_state=opt_state,
            step=state.step + 1,
        )
        metrics = StepMetrics(
            loss_mean=accum.loss_mean,
            token_count=accum.token_count,
            effective_batch_size=accum.effective_batch_size,
        )

        return new_state, metrics

    return update_step_inner


def make_streaming_update_steps(
    *,
    model_apply: Callable[[Any, Any], Any],
    optimizer: optax.GradientTransformation,
):
    """Build memory-minimal streaming accumulation functions.

    The caller runs:

        accum = init_accum(state.params)
        accum = accumulate_micro_batch(state.params, accum, batch_1)
        ...
        state, metrics = apply_accumulated_update(state, accum)

    Only ``apply_accumulated_update`` calls the optimizer, so an accumulation
    window still performs one parameter update.
    """
    grad_loss_sum_fn = make_grad_loss_sum_fn(model_apply=model_apply)

    def init_accum(params: Any) -> GradAccumState:
        return init_grad_accum_state(params)

    def accumulate_micro_batch(
        params: Any,
        accum: GradAccumState,
        batch: Batch,
    ) -> GradAccumState:
        return accumulate_one_token_normalized_grad(
            params=params,
            accum=accum,
            micro_batch=batch,
            grad_loss_sum_fn=grad_loss_sum_fn,
        )

    def apply_accumulated_update(
        state: TrainState,
        accum: GradAccumState,
    ) -> tuple[TrainState, StepMetrics]:
        result = finalize_token_normalized_grads(accum)
        updates, opt_state = optimizer.update(
            result.grads,
            state.opt_state,
            state.params,
        )
        params = optax.apply_updates(state.params, updates)
        new_state = TrainState(
            params=params,
            opt_state=opt_state,
            step=state.step + 1,
        )
        metrics = StepMetrics(
            loss_mean=result.loss_mean,
            token_count=result.token_count,
            effective_batch_size=result.effective_batch_size,
        )
        return new_state, metrics

    return init_accum, accumulate_micro_batch, apply_accumulated_update


def update_step(
    state: TrainState,
    micro_batches: Iterable[Batch],
    *,
    model_apply: Callable[[Any, Any], Any],
    optimizer: optax.GradientTransformation,
) -> tuple[TrainState, StepMetrics]:
    """Apply one optimizer update after one accumulation window.

    For compiled training, prefer make_update_step(...), then pass the returned
    function to runtime.compile_callable().
    """
    step_fn = make_update_step(
        model_apply=model_apply,
        optimizer=optimizer,
    )
    return step_fn(state, micro_batches)

__all__ = [
    "make_grad_loss_sum_fn",
    "make_streaming_update_steps",
    "make_update_step",
    "update_step",
]
