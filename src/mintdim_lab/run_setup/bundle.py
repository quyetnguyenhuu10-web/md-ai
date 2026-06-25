"""Training bundle assembly for a prepared run.

The run setup boundary is allowed to connect model, tokenizer, corpus config,
and trainer primitives. Trainer modules own optimizer, loss, update, and loop
semantics.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import optax

from mintdim_lab.model import Transformer, load_text_config_yaml
from mintdim_lab.run_setup.resolver import (
    config_override_or_training_value,
    resolve_config_path,
    resolve_training_config_paths,
)
from mintdim_lab.tokenizer.config import load_tokenizer_config_yaml
from mintdim_lab.trainer.optimizer import build_optimizer, learning_rate_schedule

if TYPE_CHECKING:
    from mintdim_lab.model import TransformerConfig
    from mintdim_lab.tokenizer.config import TokenizerConfig
    from mintdim_lab.trainer.config import TrainingConfig, UnitReadConfig
    from mintdim_lab.trainer.state import TrainState


@dataclasses.dataclass(frozen=True)
class TrainingBundle:
    """Resolved objects needed before entering the training loop."""

    training: TrainingConfig
    tokenizer: TokenizerConfig
    model_config: TransformerConfig
    unit_read: UnitReadConfig
    model: Transformer
    optimizer: optax.GradientTransformation
    update_step: Any
    init_accum: Any
    accumulate_micro_batch: Any
    apply_accumulated_update: Any


def build_model_apply(model: Transformer):
    """Return model_apply(params, input_ids) expected by training loss."""

    def model_apply(params: Any, input_ids: Any) -> Any:
        return model.apply(
            {"params": params},
            jnp.asarray(input_ids, dtype=jnp.int32),
        )

    return model_apply


def init_train_state(
    *,
    model: Transformer,
    optimizer: optax.GradientTransformation,
    rng: Any,
    input_ids: Any,
) -> TrainState:
    """Initialize Transformer params and optimizer state from a real input shape."""
    from mintdim_lab.trainer.state import TrainState

    tokens = jnp.asarray(input_ids, dtype=jnp.int32)
    variables = model.init(rng, tokens)
    params = variables["params"]
    opt_state = optimizer.init(params)

    return TrainState(
        params=params,
        opt_state=opt_state,
        step=0,
    )


def load_training_bundle(
    *,
    repo: str | Path = ".",
    training_config: str | Path = "recipes/train/cpu/linear_equation_tiny_cpu.yaml",
    tokenizer_config: str | Path | None = None,
    model_config: str | Path | None = None,
    unit_read_config: str | Path | None = None,
) -> TrainingBundle:
    """Load configs, construct model/optimizer, and expose unit_read config."""
    from mintdim_lab.trainer.config import (
        load_training_config_yaml,
        load_unit_read_config_yaml,
        validate_units_within_max_seq_len,
    )
    from mintdim_lab.trainer.update_step import (
        make_streaming_update_steps,
        make_update_step,
    )

    repo_path = Path(repo)
    training_config_path = resolve_config_path(repo_path, training_config)

    training = load_training_config_yaml(training_config_path)
    paths = resolve_training_config_paths(
        repo=repo_path,
        training_config=training_config_path,
        training=training,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        unit_read_config=unit_read_config,
    )

    tokenizer = load_tokenizer_config_yaml(paths.tokenizer_config)
    resolved_model_config = load_text_config_yaml(
        paths.model_config,
        num_embed=tokenizer.vocab_size,
    )

    unit_read = load_unit_read_config_yaml(paths.unit_read_config)
    validate_units_within_max_seq_len(
        unit_read,
        max_seq_len=resolved_model_config.max_seq_len,
    )

    model = Transformer(resolved_model_config)
    optimizer = build_optimizer(training)
    model_apply = build_model_apply(model)
    update_step = make_update_step(
        model_apply=model_apply,
        optimizer=optimizer,
    )
    init_accum, accumulate_micro_batch, apply_accumulated_update = make_streaming_update_steps(
        model_apply=model_apply,
        optimizer=optimizer,
    )

    return TrainingBundle(
        training=training,
        tokenizer=tokenizer,
        model_config=resolved_model_config,
        unit_read=unit_read,
        model=model,
        optimizer=optimizer,
        update_step=update_step,
        init_accum=init_accum,
        accumulate_micro_batch=accumulate_micro_batch,
        apply_accumulated_update=apply_accumulated_update,
    )


__all__ = [
    "TrainingBundle",
    "build_model_apply",
    "build_optimizer",
    "config_override_or_training_value",
    "init_train_state",
    "learning_rate_schedule",
    "load_training_bundle",
    "resolve_config_path",
    "resolve_training_config_paths",
]
