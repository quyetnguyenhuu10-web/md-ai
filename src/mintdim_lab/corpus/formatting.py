"""Text formatting helpers shared by corpus, chat, and inference paths."""

from __future__ import annotations

PROMPT_PREFIX = "#Prompt_user:"
ANSWER_PREFIX = "#Answer:"


def prompt_answer_prefix(prompt: str = "") -> str:
    """Return the model prompt ending immediately before answer generation."""
    text = str(prompt)
    if prompt_already_has_answer_prefix(text):
        return text
    if not text:
        return ANSWER_PREFIX + "\n"
    return PROMPT_PREFIX + " " + text.rstrip() + "\n" + ANSWER_PREFIX + "\n"


def prompt_already_has_answer_prefix(prompt: str) -> bool:
    """Return whether a prompt already ends with the answer prefix."""
    return str(prompt).rstrip().lower().endswith(ANSWER_PREFIX.lower())


__all__ = [
    "ANSWER_PREFIX",
    "PROMPT_PREFIX",
    "prompt_already_has_answer_prefix",
    "prompt_answer_prefix",
]
