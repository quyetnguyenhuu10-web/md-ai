from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from mintdim_lab.evaluator.data import Example
from mintdim_lab.evaluator.template import (
    BenchmarkTemplate,
    encode_benchmark_prompt,
    render_template_text,
    target_text,
)
from mintdim_lab.inference.kv_cache import make_kv_cache
from mintdim_lab.tokenizer import Tokenizer
from mintdim_lab.tokenizer.rules import pad_token_ids


@dataclasses.dataclass(frozen=True)
class Prediction:
    example: Example
    prediction: str
    correct: bool


def generate_predictions(
    *,
    model: Any,
    params: Any,
    config: Any,
    tokenizer: Tokenizer,
    examples: list[Example],
    prompt_template: BenchmarkTemplate,
    batch_size: int,
    max_new_tokens: int,
    progress: Callable[[int, int, int], None] | None = None,
    fixed_seq_len: int | None = None,
    scoring_flags: dict[str, bool] | None = None,
    stop_at_target_length: bool = False,
) -> list[Prediction]:
    import jax
    import jax.numpy as jnp

    from mintdim_lab.evaluator.scoring import normalize_for_exact

    @jax.jit
    def prefill_with_kv_cache(params_arg: Any, token_batch: Any, cache: Any) -> Any:
        return model.apply(
            {"params": params_arg},
            token_batch,
            cache=cache,
            cache_index=jnp.array(0, dtype=jnp.int32),
            return_cache=True,
        )

    @jax.jit
    def decode_one_token_with_kv_cache(
        params_arg: Any,
        token_batch: Any,
        cache: Any,
        cache_index: Any,
    ) -> Any:
        return model.apply(
            {"params": params_arg},
            token_batch,
            cache=cache,
            cache_index=cache_index,
            return_cache=True,
        )

    total_examples = len(examples)
    output: list[Prediction | None] = [None] * total_examples
    buffers: dict[int, list[tuple[int, Example, list[int]]]] = defaultdict(list)
    measured = 0
    correct_so_far = 0

    def process_batch(
        prompt_len: int,
        batch_rows: list[tuple[int, Example, list[int]]],
    ) -> None:
        nonlocal correct_so_far
        nonlocal measured

        real_batch_size = len(batch_rows)

        if fixed_seq_len is None:
            prompt_batch_ids = [item[2] for item in batch_rows]
            cache_len = min(
                int(config.max_seq_len),
                int(prompt_len) + int(max_new_tokens),
            )
            prompt_logits_index = -1
        else:
            cache_len = int(fixed_seq_len)
            if cache_len > int(config.max_seq_len):
                raise ValueError(
                    f"fixed_seq_len {cache_len} > model max_seq_len {config.max_seq_len}"
                )
            if int(prompt_len) >= cache_len:
                raise ValueError(
                    f"Prompt has {prompt_len} tokens, fixed_seq_len is {cache_len}: "
                    f"{batch_rows[0][1].prompt!r}"
                )
            prompt_batch_ids = [
                pad_token_ids(item[2], cache_len, pad_id=int(tokenizer.pad_id))
                for item in batch_rows
            ]
            prompt_logits_index = int(prompt_len) - 1

        prompt_batch = jnp.asarray(prompt_batch_ids, dtype=jnp.int32)

        cache = make_kv_cache(
            config,
            batch_size=real_batch_size,
            max_seq_len=cache_len,
        )

        prompt_output = prefill_with_kv_cache(
            params,
            prompt_batch,
            cache,
        )

        cache = prompt_output.cache
        logits = prompt_output.logits[:, prompt_logits_index, :]

        cache_index = int(prompt_len)
        finished = [False] * real_batch_size
        generated: list[list[int]] = [[] for _ in range(real_batch_size)]
        target_lengths = [
            len(tokenizer.encode(target_text(example, prompt_template), add_eos=False))
            for _original_index, example, _prompt_ids in batch_rows
        ]

        for _step in range(int(max_new_tokens)):
            if cache_index >= cache_len:
                break

            ids = list(map(int, jax.device_get(jnp.argmax(logits, axis=-1))))

            step_ids: list[int] = []

            for row_index, token_id in enumerate(ids):
                if finished[row_index]:
                    step_ids.append(int(tokenizer.eos_id))
                    continue

                if token_id == int(tokenizer.eos_id):
                    finished[row_index] = True
                    step_ids.append(int(tokenizer.eos_id))
                else:
                    generated[row_index].append(token_id)
                    step_ids.append(token_id)
                    if (
                        stop_at_target_length
                        and len(generated[row_index]) >= target_lengths[row_index]
                    ):
                        finished[row_index] = True

            if all(finished):
                break

            step_tokens = jnp.asarray(step_ids, dtype=jnp.int32)[:, None]

            step_output = decode_one_token_with_kv_cache(
                params,
                step_tokens,
                cache,
                jnp.asarray(cache_index, dtype=jnp.int32),
            )

            cache = step_output.cache
            logits = step_output.logits[:, -1, :]
            cache_index += 1

        for row_index, (original_index, example, _prompt_ids) in enumerate(batch_rows):
            prediction = tokenizer.decode(generated[row_index])
            answer = target_text(example, prompt_template)
            correct = normalize_for_exact(prediction, flags=scoring_flags) == normalize_for_exact(
                answer, flags=scoring_flags
            )

            output[original_index] = Prediction(
                example=example,
                prediction=prediction,
                correct=correct,
            )
            if correct:
                correct_so_far += 1

        measured += len(batch_rows)
        if progress is not None:
            progress(measured, total_examples, correct_so_far)

    for index, example in enumerate(examples):
        prompt_ids = encode_benchmark_prompt(tokenizer, example, prompt_template)

        if tokenizer.unk_id in prompt_ids:
            rendered = render_template_text(
                sequence=prompt_template.sequence,
                fields=example.fields,
                stop_before_field=prompt_template.input_until_field,
            )
            raise ValueError(f"Prompt produced <unk>: {rendered!r}")

        if len(prompt_ids) >= int(config.max_seq_len):
            rendered = render_template_text(
                sequence=prompt_template.sequence,
                fields=example.fields,
                stop_before_field=prompt_template.input_until_field,
            )
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, max_seq_len is "
                f"{config.max_seq_len}: {rendered!r}"
            )

        prompt_len = len(prompt_ids)
        buffer = buffers[prompt_len]
        buffer.append((index, example, prompt_ids))

        prepared = index + 1
        if (
            progress is not None
            and measured == 0
            and (prepared == total_examples or prepared % 10_000 == 0)
        ):
            progress(prepared, total_examples, -1)

        if len(buffer) >= int(batch_size):
            process_batch(prompt_len, buffer)
            buffers[prompt_len] = []

    for prompt_len in sorted(buffers):
        buffer = buffers[prompt_len]
        if buffer:
            process_batch(prompt_len, buffer)

    return [item for item in output if item is not None]
