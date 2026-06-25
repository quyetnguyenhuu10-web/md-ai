from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from mintdim_lab.corpus.batch import stack_batches
from mintdim_lab.trainer.gradient_accumulation import accumulate_stacked_token_normalized_grads
from mintdim_lab.trainer.state import Batch, MicroStats, TrainState
from mintdim_lab.trainer.update_step import make_streaming_update_steps


def test_stacked_accumulation_uses_lax_scan_and_preserves_effective_batch():
    params = {"w": jnp.asarray(2.0, dtype=jnp.float32)}
    batches = stack_batches(
        [
            Batch(
                input_ids=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
                target_ids=jnp.asarray([[0, 0], [0, 0]], dtype=jnp.int32),
                target_mask=jnp.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=jnp.float32),
            ),
            Batch(
                input_ids=jnp.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32),
                target_ids=jnp.asarray([[0, 0], [0, 0]], dtype=jnp.int32),
                target_mask=jnp.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=jnp.float32),
            ),
        ]
    )

    def grad_loss_sum_fn(params, batch):
        loss_sum = params["w"] * jnp.sum(batch.input_ids * batch.target_mask)
        token_count = jnp.sum(batch.target_mask)
        grads = {"w": jnp.sum(batch.input_ids * batch.target_mask)}
        return (loss_sum, MicroStats(loss_sum=loss_sum, token_count=token_count)), grads

    result = accumulate_stacked_token_normalized_grads(
        params=params,
        micro_batches=batches,
        grad_loss_sum_fn=grad_loss_sum_fn,
    )

    assert int(result.effective_batch_size) == 4
    assert float(result.token_count) == 8.0
    assert float(result.grads["w"]) == 4.5
    assert "scan" in str(
        jax.make_jaxpr(
            lambda p, b: (
                accumulate_stacked_token_normalized_grads(
                    params=p,
                    micro_batches=b,
                    grad_loss_sum_fn=grad_loss_sum_fn,
                ).loss_mean
            )
        )(params, batches)
    )


def test_streaming_accumulation_updates_optimizer_once_after_window():
    def model_apply(params, input_ids):
        x = input_ids.astype(jnp.float32)
        positive = params["w"] * x
        negative = -positive
        return jnp.stack([negative, positive], axis=-1)

    optimizer = optax.sgd(learning_rate=0.01)
    params = {"w": jnp.asarray(0.1, dtype=jnp.float32)}
    state = TrainState(
        params=params,
        opt_state=optimizer.init(params),
        step=0,
    )
    init_accum, accumulate_micro_batch, apply_accumulated_update = make_streaming_update_steps(
        model_apply=model_apply,
        optimizer=optimizer,
    )
    batches = [
        Batch(
            input_ids=jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
            target_ids=jnp.asarray([[1, 1], [1, 1]], dtype=jnp.int32),
            target_mask=jnp.asarray([[1, 1], [1, 1]], dtype=jnp.float32),
        ),
        Batch(
            input_ids=jnp.asarray([[5, 6], [7, 8]], dtype=jnp.int32),
            target_ids=jnp.asarray([[1, 1], [1, 1]], dtype=jnp.int32),
            target_mask=jnp.asarray([[1, 1], [1, 1]], dtype=jnp.float32),
        ),
    ]

    accum = init_accum(state.params)
    for batch in batches:
        accum = accumulate_micro_batch(state.params, accum, batch)
        assert state.step == 0

    new_state, metrics = apply_accumulated_update(state, accum)

    assert new_state.step == 1
    assert int(metrics.effective_batch_size) == 4
    assert float(metrics.token_count) == 8.0
    assert float(new_state.params["w"]) != float(state.params["w"])
