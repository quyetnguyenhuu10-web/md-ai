from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import optax
import pytest

from mintdim_lab.run_setup.bundle import load_training_bundle
from mintdim_lab.trainer.optimizer import learning_rate_schedule
from mintdim_lab.trainer.state import Batch, TrainState
from mintdim_lab.trainer.update_step import make_streaming_update_steps

REPO = Path(__file__).resolve().parents[2]
TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def test_training_bundle_and_schedule_contract():
    bundle = load_training_bundle(repo=REPO, model_config=TEST_MODEL_CONFIG)
    optim = bundle.training.optimization
    schedule = learning_rate_schedule(bundle.training)

    assert bundle.training.model_config.endswith(".yaml")
    assert bundle.training.tokenizer_config == "recipes/tokenizers/byte_bpe_512.yaml"
    assert (
        bundle.training.unit_read_config
        == "recipes/corpus/linear_equation_unit96/unit_read.yaml"
    )
    assert bundle.training.max_steps == 600
    assert bundle.training.save_every == 100
    assert bundle.training.checkpoint_max_to_keep == 15
    assert bundle.training.runtime.name == "cpu"
    assert bundle.training.runtime.device_index == 0
    assert bundle.training.runtime.global_device is False
    assert bundle.training.runtime.compile_update is True
    assert bundle.model_config.num_embed == bundle.tokenizer.vocab_size
    assert bundle.model_config.max_seq_len == 128
    assert bundle.model_config.num_layers == 2
    assert bundle.model_config.embed_dim == 32
    assert tuple(entry.unit for entry in bundle.unit_read.queue) == (96,)
    assert tuple(entry.batch for entry in bundle.unit_read.queue) == (50,)
    assert tuple(entry.accum for entry in bundle.unit_read.queue) == (30,)

    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(1)) == pytest.approx(
        optim.learning_rate / optim.warmup_steps,
        rel=1.0e-5,
    )
    assert float(schedule(optim.warmup_steps)) == pytest.approx(
        optim.learning_rate,
        rel=1.0e-5,
    )
    assert float(schedule(bundle.training.max_steps)) == pytest.approx(
        optim.learning_rate,
        rel=1.0e-5,
    )
    assert float(schedule(bundle.training.max_steps + 1)) == pytest.approx(
        optim.learning_rate * optim.min_lr_ratio,
        rel=1.0e-5,
    )


def test_streaming_gradient_accumulation_updates_once_per_window_contract():
    def model_apply(params, input_ids):
        x = input_ids.astype(jnp.float32)
        positive = params["w"] * x
        negative = -positive
        return jnp.stack([negative, positive], axis=-1)

    optimizer = optax.sgd(learning_rate=0.01)
    params = {"w": jnp.asarray(0.1, dtype=jnp.float32)}
    state = TrainState(params=params, opt_state=optimizer.init(params), step=0)
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
