from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mintdim_lab.corpus.batch import from_unit_read, to_jax_batch
from mintdim_lab.run_setup.bundle import build_model_apply, init_train_state, load_training_bundle
from mintdim_lab.trainer.loop import unit_read_records
from mintdim_lab.trainer.optimizer import build_optimizer, learning_rate_schedule

TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def _real_unit_read_batch():
    repo = Path(__file__).resolve().parents[2]
    bundle = load_training_bundle(repo=repo, model_config=TEST_MODEL_CONFIG)

    records = unit_read_records(**bundle.unit_read.to_unit_read_kwargs())
    record = next(records)
    batch = to_jax_batch(from_unit_read(record))

    return bundle, batch


def test_load_training_bundle_connects_configs_model_optimizer_and_unit_read():
    repo = Path(__file__).resolve().parents[2]

    bundle = load_training_bundle(repo=repo, model_config=TEST_MODEL_CONFIG)

    assert bundle.training.model_config.endswith(".yaml")
    assert bundle.training.tokenizer_config == "recipes/tokenizers/byte_bpe_512.yaml"
    assert (
        bundle.training.unit_read_config
        == "recipes/corpus/linear_equation_unit96/unit_read.yaml"
    )
    assert bundle.tokenizer.path == "./data_store/tokenizers/byte_bpe_512/tokenizer.json"
    assert bundle.model_config.num_embed == bundle.tokenizer.vocab_size
    assert bundle.model_config.max_seq_len == 128
    assert bundle.model_config.num_layers == 2
    assert bundle.model_config.embed_dim == 32
    assert tuple(entry.unit for entry in bundle.unit_read.queue) == (96,)
    assert tuple(entry.batch for entry in bundle.unit_read.queue) == (50,)
    assert (
        bundle.unit_read.queue[0].path
        == "./data_store/packed/linear_equation_unit96/unit_96"
    )
    assert bundle.unit_read.queue[0].batch == 50
    assert bundle.unit_read.queue[0].accum == 30
    assert bundle.unit_read.sequence_template == (
        "prompt",
        "token_template",
        "target",
        "token_template",
    )
    assert bundle.unit_read.target_fields == ("target", "token_template")
    assert bundle.unit_read.target_index == (2, 3)
    assert callable(bundle.update_step)


def test_learning_rate_schedule_returns_finite_nonnegative_values():
    repo = Path(__file__).resolve().parents[2]
    bundle = load_training_bundle(repo=repo, model_config=TEST_MODEL_CONFIG)

    schedule = learning_rate_schedule(bundle.training)
    values = [
        float(schedule(0)),
        float(schedule(1)),
        float(schedule(max(1, bundle.training.max_steps // 2))),
        float(schedule(bundle.training.max_steps)),
    ]

    assert all(math.isfinite(value) for value in values)
    assert all(value >= 0.0 for value in values)


def test_learning_rate_schedule_uses_one_based_training_steps():
    repo = Path(__file__).resolve().parents[2]
    bundle = load_training_bundle(repo=repo, model_config=TEST_MODEL_CONFIG)

    schedule = learning_rate_schedule(bundle.training)
    optim = bundle.training.optimization
    peak = optim.learning_rate

    assert float(schedule(1)) == pytest.approx(peak / optim.warmup_steps, rel=1e-5)
    assert float(schedule(optim.warmup_steps)) == pytest.approx(peak)
    assert float(schedule(optim.decay_start_step)) == pytest.approx(peak)
    assert float(schedule(bundle.training.max_steps)) == pytest.approx(peak, rel=1e-5)
    assert float(schedule(bundle.training.max_steps + 1)) == pytest.approx(
        peak * optim.min_lr_ratio,
        rel=1e-5,
    )


def test_optimizer_first_update_uses_first_warmup_learning_rate():
    repo = Path(__file__).resolve().parents[2]
    bundle = load_training_bundle(repo=repo, model_config=TEST_MODEL_CONFIG)
    optimizer = build_optimizer(bundle.training)
    params = {"weight": jnp.asarray(0.0)}
    opt_state = optimizer.init(params)

    updates, _ = optimizer.update({"weight": jnp.asarray(1.0)}, opt_state, params)

    assert abs(float(updates["weight"])) == pytest.approx(
        bundle.training.optimization.learning_rate / bundle.training.optimization.warmup_steps,
        rel=1e-5,
    )


def test_real_unit_read_reads_math_bin_batch():
    bundle, batch = _real_unit_read_batch()

    first_entry = bundle.unit_read.queue[0]

    assert batch.input_ids.shape == (first_entry.batch, first_entry.unit)
    assert batch.target_ids.shape == (first_entry.batch, first_entry.unit)
    assert batch.target_mask.shape == (first_entry.batch, first_entry.unit)
    assert int(batch.target_mask.sum()) > 0
    assert bundle.unit_read.target_index == (2, 3)

    mask = np.asarray(batch.target_mask).astype(bool)
    targets = np.asarray(batch.target_ids)
    assert int((targets[mask] == bundle.tokenizer.eos_id).sum()) == first_entry.batch


def test_init_train_state_from_real_unit_read_batch():
    bundle, batch = _real_unit_read_batch()

    state = init_train_state(
        model=bundle.model,
        optimizer=bundle.optimizer,
        rng=jax.random.PRNGKey(0),
        input_ids=batch.input_ids,
    )

    assert state.step == 0
    assert "input_embedding" in state.params
    assert state.opt_state is not None

    logits = build_model_apply(bundle.model)(state.params, batch.input_ids)
    assert logits.shape[:2] == batch.input_ids.shape
    assert logits.shape[-1] == bundle.model_config.num_embed


def test_one_real_optimizer_update_from_unit_read_bin():
    bundle, batch = _real_unit_read_batch()

    state = init_train_state(
        model=bundle.model,
        optimizer=bundle.optimizer,
        rng=jax.random.PRNGKey(0),
        input_ids=batch.input_ids,
    )

    new_state, metrics = bundle.update_step(state, [batch])

    assert new_state.step == 1
    assert int(metrics.effective_batch_size) == batch.input_ids.shape[0]
    assert float(metrics.token_count) > 0.0
    assert math.isfinite(float(metrics.loss_mean))

    old_leaves = jax.tree_util.tree_leaves(state.params)
    new_leaves = jax.tree_util.tree_leaves(new_state.params)

    assert len(old_leaves) == len(new_leaves)
    assert [leaf.shape for leaf in old_leaves] == [leaf.shape for leaf in new_leaves]


def test_build_optimizer_is_usable_with_initialized_state():
    bundle, batch = _real_unit_read_batch()

    optimizer = build_optimizer(bundle.training)
    state = init_train_state(
        model=bundle.model,
        optimizer=optimizer,
        rng=jax.random.PRNGKey(123),
        input_ids=batch.input_ids,
    )

    assert state.step == 0
    assert state.opt_state is not None


def test_load_training_bundle_accepts_explicit_config_overrides():
    repo = Path(__file__).resolve().parents[2]

    bundle = load_training_bundle(
        repo=repo,
        training_config="recipes/train/cpu/linear_equation_tiny_cpu.yaml",
        model_config=TEST_MODEL_CONFIG,
        tokenizer_config="recipes/tokenizers/byte_bpe_512.yaml",
        unit_read_config="recipes/corpus/linear_equation_unit96/unit_read.yaml",
    )

    assert bundle.training.model_config.endswith(".yaml")
    assert bundle.training.tokenizer_config == "recipes/tokenizers/byte_bpe_512.yaml"
    assert (
        bundle.training.unit_read_config
        == "recipes/corpus/linear_equation_unit96/unit_read.yaml"
    )
    assert bundle.model_config.num_embed == bundle.tokenizer.vocab_size
    assert bundle.model_config.num_layers == 2
