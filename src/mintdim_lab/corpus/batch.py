"""Batch conversion utilities for packed corpus records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import jax.numpy as jnp


class Batch(NamedTuple):
    """Next-token training batch.

    input_ids:
        Token ids used as model input, shape [B, T].

    target_ids:
        Token ids predicted by each input position, shape [B, T].

    target_mask:
        Boolean or numeric mask, shape [B, T]. True/nonzero positions are
        learned tokens. False/zero positions are ignored.
    """

    input_ids: Any
    target_ids: Any
    target_mask: Any


def from_unit_read(record: Any) -> Batch:
    """Create a Batch from one public MintDim unit-read batch."""
    if isinstance(record, Mapping):
        input_ids = record["input_ids"]
        target_ids = record["target_ids"]
        target_mask = record["target_mask"]
    else:
        input_ids = record.input_ids
        target_ids = record.target_ids
        target_mask = record.target_mask

    return Batch(
        input_ids=input_ids,
        target_ids=target_ids,
        target_mask=target_mask,
    )


def to_jax_batch(batch: Batch) -> Batch:
    """Convert batch arrays to JAX arrays."""
    return Batch(
        input_ids=jnp.asarray(batch.input_ids),
        target_ids=jnp.asarray(batch.target_ids),
        target_mask=jnp.asarray(batch.target_mask),
    )


def stack_batches(batches: Sequence[Batch]) -> Batch:
    """Stack micro-batches into an accumulation window with shape [A, B, T]."""
    if not batches:
        raise ValueError("stack_batches requires at least one batch.")
    return Batch(
        input_ids=jnp.stack([jnp.asarray(batch.input_ids) for batch in batches], axis=0),
        target_ids=jnp.stack([jnp.asarray(batch.target_ids) for batch in batches], axis=0),
        target_mask=jnp.stack([jnp.asarray(batch.target_mask) for batch in batches], axis=0),
    )


__all__ = [
    "Batch",
    "from_unit_read",
    "stack_batches",
    "to_jax_batch",
]
