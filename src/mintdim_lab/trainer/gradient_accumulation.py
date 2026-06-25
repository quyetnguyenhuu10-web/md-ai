"""Token-normalized gradient accumulation.

For K micro-batches, this module implements:

    final_grads = sum(grad(loss_sum_i)) / sum(token_count_i)
    loss_mean   = sum(loss_sum_i)       / sum(token_count_i)

This is equivalent to one larger batch where every learned token has equal
weight. It must not average micro-batch loss_mean values directly.

Important accumulation detail
-----------------------------

The accumulator initializes ``grad_sum`` as a zero PyTree with the same
structure, shape, and dtype as the parameter tree. Gradients produced by
``value_and_grad`` have the same PyTree structure, so each micro-batch can be
added directly into that accumulator.

Conceptually, for 4 accumulation micro-batches, the gradient accumulation is:

    grad_sum = 0 + grad_1
    grad_sum = grad_1 + grad_2
    grad_sum = grad_1 + grad_2 + grad_3
    grad_sum = grad_1 + grad_2 + grad_3 + grad_4

Only after all micro-batch gradients have been added does this module divide by
the total learned-token count:

    final_grads = (grad_1 + grad_2 + grad_3 + grad_4) / total_token_count

This module does not update model parameters. It only returns the normalized
gradients. The optimizer step is responsible for applying those gradients to
``params``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import jax
import jax.numpy as jnp

from mintdim_lab.trainer.state import AccumResult, Batch, GradAccumState, MicroStats, PyTree

GradLossSumFn = Callable[[PyTree, Batch], tuple[tuple[Any, MicroStats], PyTree]]


def tree_add(a: PyTree, b: PyTree) -> PyTree:
    """Add two PyTrees leafwise."""
    return jax.tree.map(lambda x, y: x + y, a, b)


def tree_div(tree: PyTree, scalar: Any) -> PyTree:
    """Divide a PyTree by scalar leafwise."""
    return jax.tree.map(lambda x: x / scalar, tree)


def zeros_like_tree(tree: PyTree) -> PyTree:
    """Create a zero PyTree matching another PyTree."""
    return jax.tree.map(jnp.zeros_like, tree)


def safe_token_denominator(token_count: Any) -> Any:
    """Return a nonzero denominator for token-normalized math.

    A zero-token accumulation window should normally be prevented by the data
    pipeline. This guard prevents NaN/Inf from division while preserving the
    reported token_count for the caller.
    """
    one = jnp.asarray(1.0, dtype=jnp.asarray(token_count).dtype)
    return jnp.maximum(token_count, one)


def init_grad_accum_state(params: PyTree) -> GradAccumState:
    """Create an empty gradient accumulator matching ``params``."""
    return GradAccumState(
        grad_sum=zeros_like_tree(params),
        loss_sum=jnp.asarray(0.0, dtype=jnp.float32),
        token_count=jnp.asarray(0.0, dtype=jnp.float32),
        effective_batch_size=jnp.asarray(0, dtype=jnp.int32),
    )


def accumulate_one_token_normalized_grad(
    *,
    params: PyTree,
    accum: GradAccumState,
    micro_batch: Batch,
    grad_loss_sum_fn: GradLossSumFn,
) -> GradAccumState:
    """Add one micro-batch grad(loss_sum) into an accumulator."""
    (loss_sum, stats), grads_sum = grad_loss_sum_fn(params, micro_batch)
    return GradAccumState(
        grad_sum=tree_add(accum.grad_sum, grads_sum),
        loss_sum=accum.loss_sum + stats.loss_sum,
        token_count=accum.token_count + stats.token_count,
        effective_batch_size=accum.effective_batch_size
        + jnp.asarray(micro_batch.input_ids.shape[0], dtype=jnp.int32),
    )


def finalize_token_normalized_grads(accum: GradAccumState) -> AccumResult:
    """Normalize accumulated grad(loss_sum) once by total learned-token count."""
    denominator = safe_token_denominator(accum.token_count)
    return AccumResult(
        loss_mean=accum.loss_sum / denominator,
        grads=tree_div(accum.grad_sum, denominator),
        token_count=accum.token_count,
        effective_batch_size=accum.effective_batch_size,
    )


def accumulate_token_normalized_grads(
    *,
    params: PyTree,
    micro_batches: Iterable[Batch],
    grad_loss_sum_fn: GradLossSumFn,
) -> AccumResult:
    """Accumulate gradients over micro-batches by learned-token count.

    ``grad_loss_sum_fn`` must return gradients of ``loss_sum``, not gradients of
    ``loss_mean``.

    This matters because each micro-batch may contain a different number of
    learned tokens. If each micro-batch returned gradients of ``loss_mean``, then
    a small micro-batch and a large micro-batch would be weighted equally, which
    is not equivalent to one larger batch.

    The intended accumulation sequence is:

        total_grad_sum   = grad(loss_sum_1) + grad(loss_sum_2) + ...
        total_loss_sum   = loss_sum_1       + loss_sum_2       + ...
        total_token_count = token_count_1   + token_count_2    + ...

    Then this function normalizes once at the end:

        final_grads = total_grad_sum / total_token_count
        loss_mean   = total_loss_sum / total_token_count

    Parameter updates are not performed here. This function only returns
    ``AccumResult.grads`` for a later optimizer step to apply.
    """
    accum = init_grad_accum_state(params)
    seen = False

    for micro_batch in micro_batches:
        accum = accumulate_one_token_normalized_grad(
            params=params,
            accum=accum,
            micro_batch=micro_batch,
            grad_loss_sum_fn=grad_loss_sum_fn,
        )
        seen = True

    if not seen:
        raise ValueError("accumulate_token_normalized_grads requires at least one micro-batch.")

    return finalize_token_normalized_grads(accum)


def accumulate_stacked_token_normalized_grads(
    *,
    params: PyTree,
    micro_batches: Batch,
    grad_loss_sum_fn: GradLossSumFn,
) -> AccumResult:
    """Accumulate a stacked [accum, batch, unit] window with ``lax.scan``.

    This is the compiled training path. A Python ``for`` loop over a list of
    micro-batches is unrolled by ``jax.jit`` when the list length is static; a
    large ``accum`` therefore builds one large graph. ``lax.scan`` keeps the
    accumulation loop as a loop in XLA so peak memory tracks one micro-batch
    plus the gradient accumulator, not ``accum`` unrolled copies.
    """
    def scan_micro_batch(accum: GradAccumState, micro_batch: Batch):
        accum = accumulate_one_token_normalized_grad(
            params=params,
            accum=accum,
            micro_batch=micro_batch,
            grad_loss_sum_fn=grad_loss_sum_fn,
        )
        return accum, None

    accum, _ = jax.lax.scan(
        scan_micro_batch,
        init_grad_accum_state(params),
        micro_batches,
    )
    return finalize_token_normalized_grads(accum)
