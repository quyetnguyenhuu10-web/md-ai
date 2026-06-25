from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mintdim_lab.corpus.config_paths import resolve_unit_read_config_path
from mintdim_lab.trainer.config.shared import (
    expect_mapping,
    expect_sequence,
    read_yaml_mapping,
    reject_required_only_fields,
)


@dataclasses.dataclass(frozen=True)
class UnitReadQueueEntry:
    path: str
    unit: int
    batch: int
    accum: int

    def to_entry(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "unit": self.unit,
            "batch": self.batch,
            "accum": self.accum,
        }

    def validate(self) -> None:
        if not self.path:
            raise ValueError("unit_read queue path must not be empty")
        if self.unit <= 0:
            raise ValueError("unit_read queue unit must be positive")
        if self.batch <= 0:
            raise ValueError("unit_read queue batch must be positive")
        if self.accum <= 0:
            raise ValueError("unit_read queue accum must be positive")


@dataclasses.dataclass(frozen=True)
class UnitReadLayoutConfig:
    sequence_template: tuple[str, ...]

    def validate(self) -> None:
        if not self.sequence_template:
            raise ValueError("unit_read layout.sequence_template must not be empty")
        if any(not str(segment) for segment in self.sequence_template):
            raise ValueError("unit_read layout.sequence_template entries must not be empty")


@dataclasses.dataclass(frozen=True)
class UnitReadTargetConfig:
    fields: tuple[str, ...]
    index: tuple[int, ...]
    ignore_id: int

    def validate(self, *, sequence_template: Sequence[str]) -> None:
        if not self.fields:
            raise ValueError("unit_read target.fields must not be empty")
        if len(self.fields) != len(self.index):
            raise ValueError(
                "unit_read target.fields and target.index must have the same length; "
                "each field is paired with the segment index it targets"
            )

        if any(not str(field) for field in self.fields):
            raise ValueError("unit_read target.fields entries must not be empty")

        layout_fields = set(sequence_template)
        missing_fields = [field for field in self.fields if field not in layout_fields]
        if missing_fields:
            raise ValueError(
                "unit_read target.fields entries must exist in layout.sequence_template: "
                f"{missing_fields}"
            )

        limit = len(tuple(sequence_template))
        invalid_index = [
            value for value in self.index if type(value) is not int or value < 0 or value >= limit
        ]
        if invalid_index:
            raise ValueError(
                "unit_read target.index entries are 0-base and must satisfy "
                f"0 <= index < {limit}; invalid: {invalid_index}"
            )

        template = tuple(sequence_template)
        mismatched_pairs = [
            f"{field}@{index} points to {template[index]!r}"
            for field, index in zip(self.fields, self.index, strict=True)
            if template[index] != field
        ]
        if mismatched_pairs:
            raise ValueError(
                "unit_read target field/index pairs must match layout.sequence_template: "
                + ", ".join(mismatched_pairs)
            )

        if type(self.ignore_id) is not int:
            raise ValueError("unit_read target.ignore_id must be an integer")


@dataclasses.dataclass(frozen=True)
class UnitReadConfig:
    queue: tuple[UnitReadQueueEntry, ...]
    layout: UnitReadLayoutConfig
    target: UnitReadTargetConfig

    def validate(self) -> None:
        if not self.queue:
            raise ValueError("unit_read queue must not be empty")

        for entry in self.queue:
            entry.validate()

        self.layout.validate()
        self.target.validate(sequence_template=self.layout.sequence_template)

    @property
    def entries(self) -> list[dict[str, Any]]:
        return [entry.to_entry() for entry in self.queue]

    @property
    def sequence_template(self) -> tuple[str, ...]:
        return self.layout.sequence_template

    @property
    def target_fields(self) -> tuple[str, ...]:
        return self.target.fields

    @property
    def target_index(self) -> tuple[int, ...]:
        return self.target.index

    @property
    def ignore_id(self) -> int:
        return self.target.ignore_id

    def to_unit_read_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by mintdim_lab.trainer.loop.unit_read_records."""
        return {
            "entries": self.entries,
            "sequence_template": self.sequence_template,
            "target_fields": self.target_fields,
            "target_index": self.target_index,
            "ignore_id": self.ignore_id,
        }


_REQUIRED_UNIT_READ_TOP_FIELDS = frozenset({"queue", "layout", "target"})
_REQUIRED_UNIT_READ_QUEUE_FIELDS = frozenset({"path", "unit", "batch", "accum"})
_REQUIRED_UNIT_READ_LAYOUT_FIELDS = frozenset({"sequence_template"})
_REQUIRED_UNIT_READ_TARGET_FIELDS = frozenset({"fields", "index", "ignore_id"})


def build_unit_read_config(
    values: Mapping[str, Any],
    *,
    label: str = "unit_read",
) -> UnitReadConfig:
    reject_required_only_fields(values, required=_REQUIRED_UNIT_READ_TOP_FIELDS, label=label)

    queue_values = values["queue"]
    if not isinstance(queue_values, list):
        raise ValueError(f"{label}.queue must be a list")

    queue: list[UnitReadQueueEntry] = []
    for entry_index, item in enumerate(queue_values):
        entry_label = f"{label}.queue[{entry_index}]"
        entry_values = expect_mapping(item, label=entry_label)
        reject_required_only_fields(
            entry_values,
            required=_REQUIRED_UNIT_READ_QUEUE_FIELDS,
            label=entry_label,
        )

        queue.append(
            UnitReadQueueEntry(
                path=str(entry_values["path"]),
                unit=int(entry_values["unit"]),
                batch=int(entry_values["batch"]),
                accum=int(entry_values["accum"]),
            )
        )

    layout_values = expect_mapping(values["layout"], label=f"{label}.layout")
    reject_required_only_fields(
        layout_values,
        required=_REQUIRED_UNIT_READ_LAYOUT_FIELDS,
        label=f"{label}.layout",
    )
    sequence_template = tuple(
        str(segment)
        for segment in expect_sequence(
            layout_values["sequence_template"],
            label=f"{label}.layout.sequence_template",
        )
    )

    target_values = expect_mapping(values["target"], label=f"{label}.target")
    reject_required_only_fields(
        target_values,
        required=_REQUIRED_UNIT_READ_TARGET_FIELDS,
        label=f"{label}.target",
    )

    target_fields = tuple(
        str(field)
        for field in expect_sequence(
            target_values["fields"],
            label=f"{label}.target.fields",
        )
    )

    target_index = tuple(
        _expect_target_index_value(value, label=f"{label}.target.index")
        for value in expect_sequence(
            target_values["index"],
            label=f"{label}.target.index",
        )
    )

    config = UnitReadConfig(
        queue=tuple(queue),
        layout=UnitReadLayoutConfig(sequence_template=sequence_template),
        target=UnitReadTargetConfig(
            fields=target_fields,
            index=target_index,
            ignore_id=int(target_values["ignore_id"]),
        ),
    )
    config.validate()
    return config


def _expect_target_index_value(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} entries must be integers")
    return value


def validate_units_within_max_seq_len(
    unit_read_config: UnitReadConfig,
    *,
    max_seq_len: int,
) -> None:
    """Validate unit_read queue units against a model context ceiling.

    max_seq_len is the model's maximum accepted context length.

    unit_read.queue[*].unit is the actual fixed sequence length produced by
    unit_build and read from .bin records. Training uses that unit length
    directly. It does not pad batches up to max_seq_len.

    Therefore this check enforces only:

        every queue entry unit <= max_seq_len

    It must not rewrite units, pad batches, or otherwise change data shape.
    """
    context_limit = int(max_seq_len)
    if context_limit <= 0:
        raise ValueError("max_seq_len must be positive")

    for index, entry in enumerate(unit_read_config.queue):
        if int(entry.unit) > context_limit:
            raise ValueError(
                f"unit_read queue[{index}].unit ({entry.unit}) must be <= "
                f"model max_seq_len ({context_limit})"
            )


def load_unit_read_config_yaml(path: str | Path) -> UnitReadConfig:
    """Load UnitReadConfig from YAML."""
    config_path = resolve_unit_read_config_path(path)
    return build_unit_read_config(
        read_yaml_mapping(config_path, purpose="unit_read"),
        label=config_path.stem,
    )


__all__ = [
    "UnitReadConfig",
    "UnitReadLayoutConfig",
    "UnitReadQueueEntry",
    "UnitReadTargetConfig",
    "build_unit_read_config",
    "load_unit_read_config_yaml",
    "validate_units_within_max_seq_len",
]
