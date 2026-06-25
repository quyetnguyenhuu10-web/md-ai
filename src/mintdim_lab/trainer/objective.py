"""Next-token masked loss.

This module owns loss math only. It does not read batches, select devices,
compile functions, accumulate gradients, or apply optimizer updates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from mintdim_lab.trainer.state import Batch, MicroStats, PyTree


def cross_entropy_per_token(logits: Any, target_ids: Any) -> Any:
    """Return per-token cross entropy with shape [B, T].

    logits:
        [B, T, vocab_size]

    target_ids:
        [B, T]

    target_ids must already be safe to gather. Masked ignore positions should
    be replaced before calling this function.
    """
    log_probs = jax_log_softmax(logits)
    gathered = jnp.take_along_axis(
        log_probs,
        target_ids[..., None],
        axis=-1,
    )
    return -jnp.squeeze(gathered, axis=-1)


def jax_log_softmax(logits: Any) -> Any:
    """Numerically stable log-softmax over the vocabulary axis."""
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    log_z = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_z


def loss_sum_and_stats(
    params: PyTree,
    batch: Batch,
    *,
    model_apply: Callable[[PyTree, Any], Any],
) -> tuple[Any, MicroStats]:
    """Compute summed masked next-token loss for one micro-batch.

    This returns loss_sum, not loss_mean.

    Gradient accumulation should accumulate grad(loss_sum) across
    micro-batches and divide once by total learned-token count.

    Masked target positions may contain ignore_id values such as -100. Those
    positions are replaced with a safe target id before gather, and the mask
    still controls which positions contribute to the loss.
    """
    logits = model_apply(params, batch.input_ids)

    mask_bool = batch.target_mask.astype(bool)
    safe_target_ids = jnp.where(
        mask_bool,
        batch.target_ids,
        jnp.zeros_like(batch.target_ids),
    )

    per_token_loss = cross_entropy_per_token(logits, safe_target_ids)
    mask = batch.target_mask.astype(per_token_loss.dtype)

    loss_sum = jnp.sum(per_token_loss * mask)
    token_count = jnp.sum(mask)

    return loss_sum, MicroStats(
        loss_sum=loss_sum,
        token_count=token_count,
    )
