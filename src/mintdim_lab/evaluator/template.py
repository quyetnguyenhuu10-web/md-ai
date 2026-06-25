from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from typing import Any

from mintdim_lab.evaluator.data import Example
from mintdim_lab.tokenizer import Tokenizer

_FIELD_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.-]*)\}")


@dataclasses.dataclass(frozen=True)
class BenchmarkTemplate:
    sequence: tuple[str, ...]
    input_until_field: str
    target_field: str

    def validate(self) -> None:
        if not self.sequence:
            raise ValueError("template.sequence must not be empty")
        if not self.input_until_field:
            raise ValueError("template.input_until.field must not be empty")
        if not self.target_field:
            raise ValueError("template.target.field must not be empty")

        referenced = set(template_field_names(self.sequence))
        if self.input_until_field not in referenced:
            raise ValueError(
                f"template.input_until.field {self.input_until_field!r} is not referenced "
                "by template.sequence"
            )
        if self.target_field not in referenced:
            raise ValueError(
                f"template.target.field {self.target_field!r} is not referenced "
                "by template.sequence"
            )

    def required_fields(self) -> tuple[str, ...]:
        values = set(template_field_names(self.sequence))
        values.add(self.input_until_field)
        values.add(self.target_field)
        return tuple(sorted(values))


def build_benchmark_template(
    *,
    sequence: Any,
    input_until: Any,
    target: Any,
) -> BenchmarkTemplate:
    if not isinstance(sequence, Sequence) or isinstance(sequence, str | bytes):
        raise ValueError("template.sequence must be a list of template strings")

    sequence_items = tuple(str(item) for item in sequence)
    input_until_field = _field_from_section(input_until, label="template.input_until")
    target_field = _field_from_section(target, label="template.target")

    result = BenchmarkTemplate(
        sequence=sequence_items,
        input_until_field=input_until_field,
        target_field=target_field,
    )
    result.validate()
    return result


def _field_from_section(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping with key 'field'")
    raw = value.get("field")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label}.field must be a non-empty string")
    return raw.strip()


def template_field_names(sequence: Sequence[str]) -> tuple[str, ...]:
    fields: list[str] = []
    seen: set[str] = set()
    for item in sequence:
        for match in _FIELD_PATTERN.finditer(str(item)):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                fields.append(name)
    return tuple(fields)


def render_template_text(
    *,
    sequence: Sequence[str],
    fields: Mapping[str, str],
    stop_before_field: str | None = None,
) -> str:
    parts: list[str] = []

    for segment in sequence:
        text = str(segment)
        pos = 0
        for match in _FIELD_PATTERN.finditer(text):
            field_name = match.group(1)
            if stop_before_field is not None and field_name == stop_before_field:
                parts.append(text[pos : match.start()])
                return "".join(parts)

            parts.append(text[pos : match.start()])
            if field_name not in fields:
                raise ValueError(f"Template references missing field: {field_name!r}")
            parts.append(str(fields[field_name]))
            pos = match.end()

        parts.append(text[pos:])

    return "".join(parts)


def target_text(example: Example, template: BenchmarkTemplate) -> str:
    return example.fields[template.target_field]


def encode_benchmark_prompt(
    tokenizer: Tokenizer,
    example: Example,
    template: BenchmarkTemplate,
) -> list[int]:
    text = render_template_text(
        sequence=template.sequence,
        fields=example.fields,
        stop_before_field=template.input_until_field,
    )
    return tokenizer.encode(text, add_eos=False)
