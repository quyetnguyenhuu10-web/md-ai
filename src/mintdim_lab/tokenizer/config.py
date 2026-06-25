"""Tokenizer YAML metadata and HuggingFace JSON inspection."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _read_yaml_mapping(path: str | Path, *, purpose: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(f"PyYAML is required to read {purpose} YAML configs.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        raise ValueError(f"{config_path} is empty.")
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level.")

    return dict(loaded)


def _reject_required_optional_fields(
    values: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str],
    label: str,
) -> None:
    allowed = required | optional

    missing = sorted(name for name in required if name not in values)
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")

    unknown = sorted(name for name in values if name not in allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer/vocab metadata used before model initialization."""

    type: str
    path: str
    vocab_size: int
    pad_id: int | None = None
    unk_id: int | None = None
    eos_id: int | None = None

    def validate(self) -> None:
        if self.type != "hf_json":
            raise ValueError("tokenizer.type must be 'hf_json'")
        if not self.path:
            raise ValueError("tokenizer.path must not be empty")
        if self.vocab_size <= 0:
            raise ValueError("tokenizer.vocab_size must be positive")

        for name, value in (
            ("pad_id", self.pad_id),
            ("unk_id", self.unk_id),
            ("eos_id", self.eos_id),
        ):
            if value is not None and int(value) < 0:
                raise ValueError(f"tokenizer.{name} must be >= 0 when set")
            if value is not None and int(value) >= self.vocab_size:
                raise ValueError(
                    f"tokenizer.{name} ({value}) must be < vocab_size ({self.vocab_size})"
                )


_REQUIRED_TOKENIZER_FIELDS = frozenset({"type", "path"})
_OPTIONAL_TOKENIZER_FIELDS = frozenset({"vocab_size", "pad_id", "unk_id", "eos_id"})


def _token_id_from_added_tokens(data: Mapping[str, Any], token_content: str) -> int | None:
    added_tokens = data.get("added_tokens")
    if not isinstance(added_tokens, list):
        return None

    for item in added_tokens:
        if not isinstance(item, Mapping):
            continue
        if item.get("content") == token_content and "id" in item:
            return int(item["id"])

    return None


def _token_id_from_model_vocab(data: Mapping[str, Any], token_content: str) -> int | None:
    model = data.get("model")
    if not isinstance(model, Mapping):
        return None

    vocab = model.get("vocab")
    if isinstance(vocab, Mapping) and token_content in vocab:
        return int(vocab[token_content])

    return None


def _token_id(data: Mapping[str, Any], token_content: str) -> int | None:
    added_id = _token_id_from_added_tokens(data, token_content)
    if added_id is not None:
        return added_id
    return _token_id_from_model_vocab(data, token_content)


def read_hf_json_tokenizer_metadata(path: str | Path) -> dict[str, int | None]:
    """Read vocab metadata from a Hugging Face tokenizer JSON file."""
    tokenizer_path = Path(path)
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))

    ids: list[int] = []

    model = data.get("model")
    if isinstance(model, Mapping):
        vocab = model.get("vocab")
        if isinstance(vocab, Mapping):
            ids.extend(int(value) for value in vocab.values())

    added_tokens = data.get("added_tokens")
    if isinstance(added_tokens, list):
        for item in added_tokens:
            if isinstance(item, Mapping) and "id" in item:
                ids.append(int(item["id"]))

    if not ids:
        raise ValueError(
            f"{tokenizer_path} does not expose token ids in model.vocab or added_tokens"
        )

    padding = data.get("padding")
    pad_id = None
    if isinstance(padding, Mapping) and padding.get("pad_id") is not None:
        pad_id = int(padding["pad_id"])
    if pad_id is None:
        pad_id = _token_id(data, "<pad>")

    return {
        "vocab_size": max(ids) + 1,
        "pad_id": pad_id,
        "unk_id": _token_id(data, "<unk>"),
        "eos_id": _token_id(data, "<eos>"),
    }


def build_tokenizer_config(
    values: Mapping[str, Any],
    *,
    label: str = "tokenizer",
) -> TokenizerConfig:
    """Build TokenizerConfig from YAML values."""
    _reject_required_optional_fields(
        values,
        required=_REQUIRED_TOKENIZER_FIELDS,
        optional=_OPTIONAL_TOKENIZER_FIELDS,
        label=label,
    )

    tokenizer_type = str(values["type"]).strip().lower()
    if tokenizer_type != "hf_json":
        raise ValueError("tokenizer.type must be 'hf_json'")

    metadata = read_hf_json_tokenizer_metadata(values["path"])

    vocab_size = int(values.get("vocab_size", metadata["vocab_size"]))
    pad_id = values.get("pad_id", metadata["pad_id"])
    unk_id = values.get("unk_id", metadata["unk_id"])
    eos_id = values.get("eos_id", metadata["eos_id"])

    config = TokenizerConfig(
        type=tokenizer_type,
        path=str(values["path"]),
        vocab_size=vocab_size,
        pad_id=None if pad_id is None else int(pad_id),
        unk_id=None if unk_id is None else int(unk_id),
        eos_id=None if eos_id is None else int(eos_id),
    )
    config.validate()
    return config


def load_tokenizer_config_yaml(path: str | Path) -> TokenizerConfig:
    """Load TokenizerConfig from YAML."""
    config_path = Path(path)
    return build_tokenizer_config(
        _read_yaml_mapping(config_path, purpose="tokenizer"),
        label=config_path.stem,
    )


__all__ = [
    "TokenizerConfig",
    "build_tokenizer_config",
    "load_tokenizer_config_yaml",
    "read_hf_json_tokenizer_metadata",
]
