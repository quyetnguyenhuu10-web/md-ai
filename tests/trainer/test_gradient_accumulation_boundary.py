from __future__ import annotations

import jax
import jax.numpy as jnp

from mintdim_lab.corpus.batch import stack_batches
from mintdim_lab.trainer.gradient_accumulation import (
    accumulate_stacked_token_normalized_grads,
    accumulate_token_normalized_grads,
    init_grad_accum_state,
)
from mintdim_lab.trainer.state import Batch, MicroStats


def _batch(values):
    arr = jnp.asarray(values, dtype=jnp.float32)
    return Batch(
        input_ids=arr,
        target_ids=jnp.zeros_like(arr, dtype=jnp.int32),
        target_mask=jnp.ones_like(arr, dtype=jnp.float32),
    )


def _grad_loss_sum_fn(params, batch):
    weighted = batch.input_ids * batch.target_mask
    loss_sum = params["w"] * jnp.sum(weighted)
    token_count = jnp.sum(batch.target_mask)
    grads = {"w": jnp.sum(weighted)}
    return (loss_sum, MicroStats(loss_sum=loss_sum, token_count=token_count)), grads


def test_gradient_accumulation_public_functions_live_in_trainer_boundary():
    assert callable(init_grad_accum_state)
    assert callable(accumulate_token_normalized_grads)
    assert callable(accumulate_stacked_token_normalized_grads)


def test_gradient_accumulation_normalizes_by_total_token_count():
    params = {"w": jnp.asarray(2.0, dtype=jnp.float32)}
    batches = [
        _batch([[1.0, 2.0], [3.0, 4.0]]),
        _batch([[5.0, 6.0], [7.0, 8.0]]),
    ]

    result = accumulate_token_normalized_grads(
        params=params,
        micro_batches=batches,
        grad_loss_sum_fn=_grad_loss_sum_fn,
    )

    assert int(result.effective_batch_size) == 4
    assert float(result.token_count) == 8.0
    assert float(result.grads["w"]) == 4.5


def test_stacked_gradient_accumulation_uses_lax_scan():
    params = {"w": jnp.asarray(2.0, dtype=jnp.float32)}
    batches = stack_batches(
        [
            _batch([[1.0, 2.0], [3.0, 4.0]]),
            _batch([[5.0, 6.0], [7.0, 8.0]]),
        ]
    )

    result = accumulate_stacked_token_normalized_grads(
        params=params,
        micro_batches=batches,
        grad_loss_sum_fn=_grad_loss_sum_fn,
    )

    assert float(result.grads["w"]) == 4.5
    assert "scan" in str(
        jax.make_jaxpr(
            lambda p, b: (
                accumulate_stacked_token_normalized_grads(
                    params=p,
                    micro_batches=b,
                    grad_loss_sum_fn=_grad_loss_sum_fn,
                ).loss_mean
            )
        )(params, batches)
    )
