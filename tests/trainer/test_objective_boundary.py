from __future__ import annotations

import jax.numpy as jnp
import pytest

from mintdim_lab.trainer.objective import (
    cross_entropy_per_token,
    jax_log_softmax,
    loss_sum_and_stats,
)
from mintdim_lab.trainer.state import Batch


def test_objective_public_functions_live_in_trainer_boundary():
    assert callable(cross_entropy_per_token)
    assert callable(jax_log_softmax)
    assert callable(loss_sum_and_stats)


def test_loss_sum_masks_ignore_targets_before_gather():
    params = {"scale": jnp.asarray(1.0)}

    def model_apply(params, input_ids):
        del params
        batch, length = input_ids.shape
        logits = jnp.zeros((batch, length, 3), dtype=jnp.float32)
        return logits.at[..., 1].set(2.0)

    batch = Batch(
        input_ids=jnp.asarray([[0, 0]], dtype=jnp.int32),
        target_ids=jnp.asarray([[1, -100]], dtype=jnp.int32),
        target_mask=jnp.asarray([[1, 0]], dtype=jnp.float32),
    )

    loss_sum, stats = loss_sum_and_stats(params, batch, model_apply=model_apply)

    assert float(stats.token_count) == 1.0
    assert float(stats.loss_sum) == pytest.approx(float(loss_sum))
    assert float(loss_sum) > 0.0
