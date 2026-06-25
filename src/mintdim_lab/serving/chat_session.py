"""Inference warm-up and stdio prompt loop."""

from __future__ import annotations

import json
import sys
from typing import Any

from mintdim_lab.serving.prompting import chat_prompt_for_model
from mintdim_lab.serving.worker_protocol import write_error, write_event

_WARMUP_PROMPTS: tuple[str, ...] = (
    "Một prompt khởi động ngắn.",
    "Một prompt khởi động đủ dài để chuẩn bị bucket prefill lớn hơn trước khi chat.",
)


def warm_up(
    *,
    generate_fn: Any,
    model: Any,
    variables: Any,
    config: Any,
    tokenizer: Any,
    rng: Any,
    raw_prompt: bool,
) -> Any:
    for prompt in _WARMUP_PROMPTS:
        model_prompt = chat_prompt_for_model(prompt, raw_prompt=raw_prompt)
        _, rng = generate_fn(
            model=model,
            variables=variables,
            config=config,
            tokenizer=tokenizer,
            prompt=model_prompt,
            max_new_tokens=2,
            temperature=0.0,
            top_k=0,
            rng=rng,
        )
    return rng


def prompt_loop(
    *,
    args: Any,
    model: Any,
    variables: Any,
    config: Any,
    tokenizer: Any,
    generate_fn: Any,
    rng: Any,
) -> None:
    raw_prompt = bool(args.raw_prompt)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            write_error(f"Invalid JSON: {exc}")
            continue

        msg_type = msg.get("type")
        if msg_type == "shutdown":
            break
        if msg_type != "prompt":
            write_error(f"Unknown command type: {msg_type!r}")
            continue

        prompt = str(msg.get("text", "")).strip()
        if not prompt:
            write_event({"event": "done"})
            continue

        model_prompt = chat_prompt_for_model(prompt, raw_prompt=raw_prompt)

        def on_token(token_text: str) -> None:
            write_event({"event": "token", "text": token_text})

        try:
            _, rng = generate_fn(
                model=model,
                variables=variables,
                config=config,
                tokenizer=tokenizer,
                prompt=model_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                rng=rng,
                on_token=on_token,
            )
            write_event({"event": "done"})
        except Exception as exc:
            write_error(f"Generation failed: {exc}")


__all__ = ["prompt_loop", "warm_up"]
