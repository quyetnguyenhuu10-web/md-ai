# Token-normalized gradient accumulation

Status: archival design note. The production training path lives in
`mintdim_lab.trainer` and the runnable configuration lives under
`recipes/train/`.

This note preserves the intended next-token training and gradient accumulation design.

The core idea is:

~~~text
final_loss =
    sum(loss_total_i)
    /
    sum(token_count_i)

final_grads =
    sum(grad(loss_total_i))
    /
    sum(token_count_i)
~~~

Where token_count_i is the number of learned tokens in micro-batch i, computed from target_mask.

The original preservation file is kept below as a Python design sketch.

~~~python
"""Next-token training design: token-normalized gradient accumulation.

This file preserves the core training idea before implementation.

Core pipeline
-------------
unit_read
    -> produces batch fields:
        input_ids
        target_ids
        target_mask

training layer
    -> converts batch arrays to JAX arrays
    -> places PyTrees on explicit runtime device
    -> runs next-token forward pass
    -> computes masked token loss
    -> accumulates gradients by learned-token count
    -> applies one optimizer update after K micro-batches

Important boundary
------------------
unit_read:
    - reads .bin records
    - creates input_ids, target_ids, target_mask
    - does not know JAX
    - does not know model/loss/optimizer/training loop

mintdim_lab.system runtime adapters:
    - require_device()
    - put_tree()
    - compile_callable()
    - sync()
    - does not know training semantics

training:
    - owns model forward
    - owns loss
    - owns grad accumulation
    - owns optimizer update
    - owns checkpoint/logging policy later

Teacher forcing rule
--------------------
For each fixed-width payload sequence:

    input_ids[t]  = payload[t]
    target_ids[t] = payload[t + 1]

The final target position uses ignore_id.

target_mask is true only when:
    - the target token belongs to a target segment
    - the target token is not pad_id
    - the target token is not ignore_id

Loss objective
--------------
The objective is average negative log likelihood per learned token:

    loss_mean = loss_sum / learned_token_count

where:

    loss_sum = sum(cross_entropy(logits, target_ids) * target_mask)

    learned_token_count = sum(target_mask)

Gradient accumulation rule
--------------------------
For K micro-batches, do NOT average micro-batch loss_mean values.

Wrong when learned-token counts differ:

    wrong_loss = (loss_mean_1 + loss_mean_2 + ... + loss_mean_K) / K

Correct batch-equivalent objective:

    final_loss =
        (loss_total_1 + loss_total_2 + ... + loss_total_K)
        /
        (token_count_1 + token_count_2 + ... + token_count_K)

Correct accumulated gradient:

    final_grads =
        (
            grad(loss_total_1)
          + grad(loss_total_2)
          + ...
          + grad(loss_total_K)
        )
        /
        (
            token_count_1
          + token_count_2
          + ...
          + token_count_K
        )

Why this is correct
-------------------
A real large batch computes one mean loss over all learned tokens.

Therefore, gradient accumulation must make every learned token have equal
weight, not every micro-batch.

If micro-batches have different target_mask counts, averaging loss_mean per
micro-batch gives too much weight to smaller micro-batches.

Implementation sketch
---------------------
This is intentionally not final production code. It records the intended math
and boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

PyTree = Any


class Batch(NamedTuple):
    """Batch produced by unit_read and converted by training layer."""

    input_ids: Any       # [B, T]
    target_ids: Any      # [B, T]
    target_mask: Any     # [B, T], bool or 0/1


class MicroStats(NamedTuple):
    """Per-micro-batch scalar stats."""

    loss_sum: Any
    token_count: Any


class AccumResult(NamedTuple):
    """Result after K micro-batches."""

    loss_mean: Any
    grads: PyTree
    token_count: Any


def cross_entropy_per_token(logits: Any, target_ids: Any) -> Any:
    """Return per-token CE with shape [B, T].

    Placeholder only.

    Expected:
        logits:     [B, T, vocab_size]
        target_ids: [B, T]

    Return:
        per_token_loss: [B, T]
    """
    raise NotImplementedError


def tree_add(a: PyTree, b: PyTree) -> PyTree:
    """Add two gradient PyTrees leafwise.

    Placeholder only. In real JAX code, use jax.tree.map.
    """
    raise NotImplementedError


def tree_div(tree: PyTree, scalar: Any) -> PyTree:
    """Divide a gradient PyTree by a scalar leafwise.

    Placeholder only. In real JAX code, use jax.tree.map.
    """
    raise NotImplementedError


def zeros_like_tree(tree: PyTree) -> PyTree:
    """Create a zero PyTree matching another PyTree.

    Placeholder only. In real JAX code, use jax.tree.map.
    """
    raise NotImplementedError


def loss_sum_fn(
    params: PyTree,
    batch: Batch,
    *,
    model_apply: Callable[[PyTree, Any], Any],
) -> tuple[Any, MicroStats]:
    """Compute summed masked next-token loss for one micro-batch.

    This function returns loss_sum, not loss_mean.

    That is deliberate. Grad accumulation should accumulate grad(loss_sum)
    and divide once by the total learned-token count across all micro-batches.
    """
    logits = model_apply(params, batch.input_ids)

    per_token_loss = cross_entropy_per_token(
        logits=logits,
        target_ids=batch.target_ids,
    )

    mask = batch.target_mask

    # Real JAX code:
    #
    #   mask = batch.target_mask.astype(per_token_loss.dtype)
    #   loss_sum = jnp.sum(per_token_loss * mask)
    #   token_count = jnp.sum(mask)
    #
    # Here this is only the intended math.
    loss_sum = "sum(per_token_loss * target_mask)"
    token_count = "sum(target_mask)"

    stats = MicroStats(
        loss_sum=loss_sum,
        token_count=token_count,
    )

    return loss_sum, stats


def accumulate_token_normalized_grads(
    *,
    params: PyTree,
    micro_batches: Iterable[Batch],
    grad_loss_sum_fn: Callable[[PyTree, Batch], tuple[tuple[Any, MicroStats], PyTree]],
) -> AccumResult:
    """Accumulate gradients across micro-batches by learned-token count.

    Required behavior for each micro-batch:

        (loss_sum, stats), grads_sum = grad_loss_sum_fn(params, micro_batch)

    where grads_sum is:

        grad(loss_sum)

    not:

        grad(loss_sum / token_count)

    Final normalization:

        final_grads = sum(grads_sum_i) / sum(token_count_i)
        loss_mean   = sum(loss_sum_i)  / sum(token_count_i)

    This makes accumulation equivalent to one real larger batch, assuming
    same params are used for all micro-batches before the optimizer update.
    """
    grad_sum = None
    total_loss_sum = 0.0
    total_token_count = 0.0

    for micro_batch in micro_batches:
        (loss_sum, stats), grads_sum = grad_loss_sum_fn(params, micro_batch)

        if grad_sum is None:
            grad_sum = zeros_like_tree(grads_sum)

        grad_sum = tree_add(grad_sum, grads_sum)
        total_loss_sum = total_loss_sum + stats.loss_sum
        total_token_count = total_token_count + stats.token_count

    final_grads = tree_div(grad_sum, total_token_count)
    loss_mean = total_loss_sum / total_token_count

    return AccumResult(
        loss_mean=loss_mean,
        grads=final_grads,
        token_count=total_token_count,
    )


def training_step_shape() -> None:
    """Intended final training step shape.

    Pseudocode:

        from mintdim_lab.system import jax_tpu as runtime

        device = runtime.require_device(index=0, local=True)

        params = runtime.put_tree(params, device)
        opt_state = runtime.put_tree(opt_state, device)

        step_fn = runtime.compile_callable(step_fn)

        for accumulation window:
            micro_batches = []

            for _ in range(accum):
                batch = unit_read.read()
                batch = convert_to_jax_arrays(batch)
                batch = runtime.put_tree(batch, device)
                micro_batches.append(batch)

            result = accumulate_token_normalized_grads(
                params=params,
                micro_batches=micro_batches,
                grad_loss_sum_fn=grad_loss_sum_fn,
            )

            updates, opt_state = optimizer.update(
                result.grads,
                opt_state,
                params,
            )

            params = apply_updates(params, updates)

            log:
                loss = result.loss_mean
                learned_tokens = result.token_count

    The optimizer update happens once per accumulation window.
    """
    raise NotImplementedError
~~~
