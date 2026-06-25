"""Adapter for the public MintDim unit-read pipeline."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

UnitReadRecord = Any


def unit_read_pipeline() -> Any:
    """Return the public `mintdim` unit-read pipeline."""
    from mintdim import pipeline

    return pipeline("unit-read")


def unit_read_records(
    *,
    entries: Sequence[dict[str, Any]],
    sequence_template: Sequence[str],
    target_fields: Sequence[str],
    target_index: Sequence[int] = (),
    ignore_id: int = -100,
) -> Iterator[UnitReadRecord]:
    """Yield records from external public `mintdim` unit-read API."""
    reader = (
        unit_read_pipeline()
        .queue(entries=[dict(entry) for entry in entries])
        .layout(sequence_template=tuple(sequence_template))
        .target(
            fields=list(target_fields),
            index=list(target_index),
            ignore_id=int(ignore_id),
        )
    )

    while True:
        yield reader.read()


__all__ = [
    "UnitReadRecord",
    "unit_read_pipeline",
    "unit_read_records",
]
