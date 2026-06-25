"""Runtime HuggingFace tokenizer JSON adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import read_hf_json_tokenizer_metadata


class Tokenizer:
    """Runtime adapter around a Hugging Face tokenizer JSON file."""

    def __init__(self, backend: Any, *, path: str | Path) -> None:
        self.backend = backend
        self.path = str(path)
        metadata = read_hf_json_tokenizer_metadata(path)
        self.vocab_size = int(metadata["vocab_size"])
        self.pad_id = 0 if metadata["pad_id"] is None else int(metadata["pad_id"])
        self.unk_id = -1 if metadata["unk_id"] is None else int(metadata["unk_id"])
        self.eos_id = -1 if metadata["eos_id"] is None else int(metadata["eos_id"])

    @property
    def token_to_id(self) -> dict[str, int]:
        getter = getattr(self.backend, "get_vocab", None)
        if getter is None:
            return {}
        return dict(getter())

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        encoding = self.backend.encode(str(text), add_special_tokens=False)
        ids = [int(value) for value in encoding.ids]
        if add_eos and self.eos_id >= 0:
            ids.append(int(self.eos_id))
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        values = [int(value) for value in ids]
        return str(self.backend.decode(values, skip_special_tokens=True))


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a runtime tokenizer from HuggingFace tokenizer JSON."""
    tokenizer_path = Path(path)
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError as exc:
        raise RuntimeError("tokenizers is required for runtime tokenizer loading.") from exc

    return Tokenizer(HFTokenizer.from_file(str(tokenizer_path)), path=tokenizer_path)


__all__ = [
    "Tokenizer",
    "load_tokenizer",
]
