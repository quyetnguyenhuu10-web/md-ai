"""Trainer optimizer and learning-rate schedule construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import optax

if TYPE_CHECKING:
    from mintdim_lab.trainer.config import TrainingConfig


def learning_rate_schedule(training: TrainingConfig) -> optax.Schedule:
    """Build an optax schedule from TrainingConfig.

    The returned schedule accepts a 1-based training step. For example, step 1
    is the first optimizer update and ``warmup_steps`` reaches the peak learning
    rate exactly at that numbered training step.
    """

    optim = training.optimization
    peak = float(optim.learning_rate)
    max_steps = max(1, int(training.max_steps))
    warmup_steps = max(0, int(optim.warmup_steps))
    decay_start_step = max(0, int(optim.decay_start_step))
    decay_start_step = max(decay_start_step, warmup_steps)
    decay_start_step = min(decay_start_step, max_steps)

    schedules: list[optax.Schedule] = []
    boundaries: list[int] = []

    if warmup_steps > 0:
        schedules.append(
            optax.linear_schedule(
                init_value=0.0,
                end_value=peak,
                transition_steps=max(1, warmup_steps),
            )
        )
        boundaries.append(warmup_steps)

    if decay_start_step > warmup_steps:
        schedules.append(optax.constant_schedule(peak))
        boundaries.append(decay_start_step)

    decay_steps = max(1, max_steps - decay_start_step)
    schedules.append(
        optax.cosine_decay_schedule(
            init_value=peak,
            decay_steps=decay_steps,
            alpha=float(optim.min_lr_ratio),
        )
    )

    if len(schedules) == 1:
        return schedules[0]

    return optax.join_schedules(
        schedules=schedules,
        boundaries=boundaries[: len(schedules) - 1],
    )


def build_optimizer(training: TrainingConfig) -> optax.GradientTransformation:
    """Build the optimizer used by the real training path."""

    optim = training.optimization
    schedule = learning_rate_schedule(training)

    def optimizer_schedule(update_count):
        """Map Optax's 0-based update count to the 1-based training step."""
        return schedule(update_count + 1)

    return optax.chain(
        optax.clip_by_global_norm(float(optim.clip_grad_norm)),
        optax.adamw(
            learning_rate=optimizer_schedule,
            b1=float(optim.beta1),
            b2=float(optim.beta2),
            eps=float(optim.adam_eps),
            weight_decay=float(optim.weight_decay),
        ),
    )

__all__ = [
    "build_optimizer",
    "learning_rate_schedule",
]
