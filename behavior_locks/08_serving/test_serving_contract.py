from __future__ import annotations

import jax
import jax.numpy as jnp

from mintdim_lab.inference.model_loader import build_model_and_generate_fn
from mintdim_lab.inference.text_generator import sample_next_token
from mintdim_lab.model import Transformer
from mintdim_lab.serving.prompting import chat_prompt_for_model


def test_chat_prompt_raw_by_default_and_formatted_when_requested():
    prompt = "Giải phương trình: 2x-2=3?"

    assert chat_prompt_for_model(prompt) == prompt
    formatted = chat_prompt_for_model(prompt, raw_prompt=False)

    assert prompt in formatted
    assert formatted != prompt


def test_temperature_zero_sampling_is_argmax_and_rng_stable():
    rng = jax.random.PRNGKey(123)
    token_id, next_rng = sample_next_token(
        jnp.asarray([0.1, 2.0, 1.5], dtype=jnp.float32),
        temperature=0.0,
        top_k=0,
        rng=rng,
    )

    assert token_id == 1
    assert bool(jnp.all(next_rng == rng))


def test_serving_runtime_builds_text_transformer_generate_pair():
    class Config:
        pass

    model, generate_fn = build_model_and_generate_fn(config=Config())

    assert isinstance(model, Transformer)
    assert callable(generate_fn)
