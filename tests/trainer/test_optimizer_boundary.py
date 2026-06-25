from __future__ import annotations

import jax.numpy as jnp
import pytest

import mintdim_lab.run_setup.bundle as run_setup_bundle
from mintdim_lab.trainer.config import OptimizationConfig, TrainingConfig
from mintdim_lab.trainer.optimizer import build_optimizer, learning_rate_schedule


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        model_config="recipes/models/test_tiny.yaml",
        tokenizer_config="recipes/tokenizers/byte_bpe_512.yaml",
        unit_read_config="recipes/corpus/linear_equation_unit96/unit_read.yaml",
        checkpoint_dir="runs/checkpoints",
        checkpoint_max_to_keep=2,
        max_steps=5,
        seed=0,
        log_every=1,
        save_every=1,
        optimization=OptimizationConfig(
            learning_rate=1.0,
            warmup_steps=2,
            decay_start_step=5,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.999,
            adam_eps=1.0e-8,
            clip_grad_norm=1.0,
            min_lr_ratio=0.1,
        ),
    )


def test_optimizer_helpers_live_in_trainer_and_are_shared_with_run_setup_bundle():
    assert run_setup_bundle.learning_rate_schedule is learning_rate_schedule
    assert run_setup_bundle.build_optimizer is build_optimizer


def test_learning_rate_schedule_keeps_one_based_training_step_contract():
    cfg = _training_config()
    schedule = learning_rate_schedule(cfg)

    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(1)) == pytest.approx(0.5)
    assert float(schedule(cfg.optimization.warmup_steps)) == pytest.approx(1.0)
    assert float(schedule(cfg.max_steps)) == pytest.approx(1.0)
    assert float(schedule(cfg.max_steps + 1)) == pytest.approx(0.1, rel=1.0e-5)


def test_build_optimizer_uses_first_warmup_learning_rate_on_first_update():
    cfg = _training_config()
    optimizer = build_optimizer(cfg)
    params = {"weight": jnp.asarray(0.0)}
    opt_state = optimizer.init(params)

    updates, _ = optimizer.update(
        {"weight": jnp.asarray(1.0)},
        opt_state,
        params,
    )

    expected = cfg.optimization.learning_rate / cfg.optimization.warmup_steps
    assert abs(float(updates["weight"])) == pytest.approx(expected, rel=1.0e-5)
