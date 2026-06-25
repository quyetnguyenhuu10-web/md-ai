from __future__ import annotations

from mintdim_lab.corpus.formatting import prompt_answer_prefix

EXIT_COMMANDS = {"/exit", "/quit"}


def chat_prompt_for_model(
    prompt: str,
    *,
    raw_prompt: bool = True,
) -> str:
    if raw_prompt:
        return prompt
    return prompt_answer_prefix(prompt=prompt)
