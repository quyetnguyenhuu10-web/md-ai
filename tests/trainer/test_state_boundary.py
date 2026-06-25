from __future__ import annotations

import jax.numpy as jnp

from mintdim_lab.corpus.batch import Batch as CorpusBatch
from mintdim_lab.trainer.state import (
    AccumResult,
    Batch,
    GradAccumState,
    MicroStats,
    StepMetrics,
    TrainState,
)


def test_trainer_state_owns_training_types_and_reexports_corpus_batch():
    assert Batch is CorpusBatch
    assert TrainState.__module__ == "mintdim_lab.trainer.state"

    batch = Batch(
        input_ids=jnp.asarray([[1, 2]], dtype=jnp.int32),
        target_ids=jnp.asarray([[2, 3]], dtype=jnp.int32),
        target_mask=jnp.asarray([[1, 1]], dtype=jnp.float32),
    )
    state = TrainState(params={"w": jnp.asarray(1.0)}, opt_state={}, step=0)

    assert batch.input_ids.shape == (1, 2)
    assert state.step == 0


def test_trainer_state_metrics_are_stable_namedtuple_contracts():
    stats = MicroStats(loss_sum=jnp.asarray(2.0), token_count=jnp.asarray(4.0))
    accum = GradAccumState(
        grad_sum={"w": jnp.asarray(1.0)},
        loss_sum=stats.loss_sum,
        token_count=stats.token_count,
        effective_batch_size=jnp.asarray(1),
    )
    result = AccumResult(
        loss_mean=jnp.asarray(0.5),
        grads=accum.grad_sum,
        token_count=accum.token_count,
        effective_batch_size=accum.effective_batch_size,
    )
    metrics = StepMetrics(
        loss_mean=result.loss_mean,
        token_count=result.token_count,
        effective_batch_size=result.effective_batch_size,
    )

    assert float(stats.loss_sum) == 2.0
    assert float(result.loss_mean) == 0.5
    assert int(metrics.effective_batch_size) == 1
