from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mintdim_lab.inference.kv_cache import make_kv_cache
from mintdim_lab.tokenizer.rules import pad_token_ids

_DECODE_APPLY_CACHE: dict[int, Any] = {}
_PREFILL_APPLY_CACHE: dict[tuple[int, int], Any] = {}


def generate(
    *,
    model: Any,
    variables: dict[str, Any],
    config: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    rng: Any,
    on_token: Callable[[str], None] | None = None,
    stop_when: Callable[[str], bool] | None = None,
) -> tuple[str, Any]:
    import jax.numpy as jnp

    prompt_ids = tokenizer.encode(prompt, add_eos=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_id]

    _validate_context_len(len(prompt_ids), config.max_seq_len)

    cache = make_kv_cache(config, batch_size=1)
    bucket_len = _prefill_bucket_len(len(prompt_ids), config.max_seq_len)
    padded_ids = pad_token_ids(prompt_ids, bucket_len, pad_id=tokenizer.pad_id)
    tokens = jnp.asarray([padded_ids], dtype=jnp.int32)

    output = _prefill_apply_for_model(model, bucket_len)(
        variables,
        tokens,
        cache,
    )

    logits = output.logits[0, len(prompt_ids) - 1]

    generated: list[int] = []
    generated_text = ""
    cache = output.cache
    cache_index = len(prompt_ids)

    for index in range(max_new_tokens):
        if cache_index >= config.max_seq_len:
            break

        token_id, rng = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        if token_id == tokenizer.eos_id:
            break

        token_text = tokenizer.decode([token_id])
        generated.append(token_id)
        generated_text += token_text

        if on_token is not None:
            on_token(token_text)

        if stop_when is not None and stop_when(generated_text):
            break

        if index == max_new_tokens - 1 or cache_index + 1 >= config.max_seq_len:
            break

        step_tokens = jnp.asarray([[token_id]], dtype=jnp.int32)
        output = _decode_apply_for_model(model)(
            variables,
            step_tokens,
            cache,
            jnp.asarray(cache_index, dtype=jnp.int32),
        )

        cache = output.cache
        logits = output.logits[0, -1]
        cache_index += 1

    return generated_text or tokenizer.decode(generated), rng


def _validate_context_len(token_count: int, max_seq_len: int) -> None:
    if token_count >= max_seq_len:
        raise ValueError(f"Prompt has {token_count} tokens, max_seq_len is {max_seq_len}.")


def _decode_apply_for_model(model: Any) -> Any:
    import jax

    key = id(model)
    cached = _DECODE_APPLY_CACHE.get(key)
    if cached is not None:
        return cached

    @jax.jit
    def apply_decode_step(
        variables: dict[str, Any],
        tokens: Any,
        cache: Any,
        cache_index: Any,
    ) -> Any:
        return model.apply(
            variables,
            tokens,
            cache=cache,
            cache_index=cache_index,
            return_cache=True,
        )

    _DECODE_APPLY_CACHE[key] = apply_decode_step
    return apply_decode_step


def _prefill_apply_for_model(model: Any, bucket_len: int) -> Any:
    import jax
    import jax.numpy as jnp

    key = (id(model), int(bucket_len))
    cached = _PREFILL_APPLY_CACHE.get(key)
    if cached is not None:
        return cached

    @jax.jit
    def apply_prefill(
        variables: dict[str, Any],
        tokens: Any,
        cache: Any,
    ) -> Any:
        return model.apply(
            variables,
            tokens,
            cache=cache,
            cache_index=jnp.array(0, dtype=jnp.int32),
            return_cache=True,
        )

    _PREFILL_APPLY_CACHE[key] = apply_prefill
    return apply_prefill


def _prefill_bucket_len(token_count: int, max_seq_len: int) -> int:
    bucket = 16
    while bucket < int(token_count):
        bucket *= 2
    return min(bucket, int(max_seq_len))


def sample_next_token(
    logits: Any,
    *,
    temperature: float,
    top_k: int,
    rng: Any,
) -> tuple[int, Any]:
    import jax
    import jax.numpy as jnp

    logits = jnp.asarray(logits, dtype=jnp.float32)
    if temperature <= 0:
        return int(jax.device_get(jnp.argmax(logits))), rng

    logits = logits / float(temperature)
    if top_k > 0:
        k = min(int(top_k), int(logits.shape[-1]))
        values, indices = jax.lax.top_k(logits, k)
        rng, subkey = jax.random.split(rng)
        selected = jax.random.categorical(subkey, values)
        return int(jax.device_get(indices[selected])), rng

    rng, subkey = jax.random.split(rng)
    token_id = jax.random.categorical(subkey, logits)
    return int(jax.device_get(token_id)), rng
