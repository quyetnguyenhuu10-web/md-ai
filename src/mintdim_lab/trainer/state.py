"""Trainer state, metrics, and accumulation data types.

These types describe the training layer contract. They do not read datasets,
select devices, compile functions, or update parameters by themselves.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from mintdim_lab.corpus.batch import Batch

PyTree = Any


class MicroStats(NamedTuple):
    """Scalar stats for one micro-batch."""

    loss_sum: Any
    token_count: Any


class AccumResult(NamedTuple):
    """Result after one gradient accumulation window."""

    loss_mean: Any
    grads: PyTree
    token_count: Any
    effective_batch_size: Any


class GradAccumState(NamedTuple):
    """Mutable-by-return gradient accumulation state for one optimizer step."""

    grad_sum: PyTree
    loss_sum: Any
    token_count: Any
    effective_batch_size: Any


class TrainState(NamedTuple):
    """Minimal train state.

    params:
        Model parameters.

    opt_state:
        Optimizer state.

    step:
        Number of optimizer updates already applied.
    """

    params: PyTree
    opt_state: PyTree
    step: int


class StepMetrics(NamedTuple):
    """Metrics emitted by one optimizer update."""

    loss_mean: Any
    token_count: Any
    effective_batch_size: Any


__all__ = [
    "AccumResult",
    "Batch",
    "GradAccumState",
    "MicroStats",
    "PyTree",
    "StepMetrics",
    "TrainState",
]
