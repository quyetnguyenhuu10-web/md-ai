"""Tokenizer vocabulary rules shared by training and runtime code."""

from __future__ import annotations

SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<eos>", "<unk>")

METADATA_TOKENS: tuple[str, ...] = (
    "#Time stamp:",
    "#Prompt_user:",
    "#Answer:",
)
REQUIRED_VOCAB_TOKENS: tuple[str, ...] = (*tuple(str(value) for value in range(10)), "=")
TOKENIZER_SPECIAL_TOKENS: tuple[str, ...] = (
    *SPECIAL_TOKENS,
    *METADATA_TOKENS,
)


def pad_token_ids(token_ids: list[int], target_len: int, *, pad_id: int) -> list[int]:
    missing = int(target_len) - len(token_ids)
    if missing <= 0:
        return token_ids
    return [*token_ids, *([int(pad_id)] * missing)]


__all__ = [
    "METADATA_TOKENS",
    "REQUIRED_VOCAB_TOKENS",
    "SPECIAL_TOKENS",
    "TOKENIZER_SPECIAL_TOKENS",
    "pad_token_ids",
]
