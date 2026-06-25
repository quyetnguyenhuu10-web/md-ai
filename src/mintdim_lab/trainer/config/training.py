from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mintdim_lab.trainer.config.shared import (
    expect_bool,
    expect_mapping,
    expect_plain_int,
    read_yaml_mapping,
    reject_required_only_fields,
)


@dataclasses.dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer and scheduler scalar settings."""

    learning_rate: float
    warmup_steps: int
    decay_start_step: int
    weight_decay: float
    beta1: float
    beta2: float
    adam_eps: float
    clip_grad_norm: float
    min_lr_ratio: float

    def validate(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("optimization.learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("optimization.warmup_steps must be >= 0")
        if self.decay_start_step < 0:
            raise ValueError("optimization.decay_start_step must be >= 0")
        if self.weight_decay < 0.0:
            raise ValueError("optimization.weight_decay must be >= 0")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("optimization.beta1 must be in [0.0, 1.0)")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("optimization.beta2 must be in [0.0, 1.0)")
        if self.adam_eps <= 0.0:
            raise ValueError("optimization.adam_eps must be positive")
        if self.clip_grad_norm <= 0.0:
            raise ValueError("optimization.clip_grad_norm must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("optimization.min_lr_ratio must be in [0.0, 1.0]")


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """JAX runtime selection for one training recipe."""

    name: str
    device_index: int
    global_device: bool
    compile_update: bool

    def validate(self) -> None:
        if self.name not in {"cpu", "gpu", "tpu"}:
            raise ValueError("runtime.name must be one of: cpu, gpu, tpu")
        if type(self.device_index) is not int:
            raise ValueError("runtime.device_index must be an integer")
        if self.device_index < 0:
            raise ValueError("runtime.device_index must be >= 0")
        if type(self.global_device) is not bool:
            raise ValueError("runtime.global_device must be a boolean")
        if type(self.compile_update) is not bool:
            raise ValueError("runtime.compile_update must be a boolean")


def _default_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        name="cpu",
        device_index=0,
        global_device=False,
        compile_update=True,
    )


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Static training-run config.

    This object does not build models, read datasets, create optimizers, call
    unit_read, or perform training math.
    """

    model_config: str
    tokenizer_config: str
    unit_read_config: str
    checkpoint_dir: str
    checkpoint_max_to_keep: int
    max_steps: int
    seed: int
    log_every: int
    save_every: int
    optimization: OptimizationConfig
    runtime: RuntimeConfig = dataclasses.field(default_factory=_default_runtime_config)

    def validate(self) -> None:
        if not self.model_config:
            raise ValueError("model_config must not be empty")
        if not self.tokenizer_config:
            raise ValueError("tokenizer_config must not be empty")
        if not self.unit_read_config:
            raise ValueError("unit_read_config must not be empty")
        if not self.checkpoint_dir:
            raise ValueError("checkpoint_dir must not be empty")
        if self.checkpoint_max_to_keep <= 0:
            raise ValueError("checkpoint_max_to_keep must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.save_every <= 0:
            raise ValueError("save_every must be positive")
        self.optimization.validate()
        self.runtime.validate()


_REQUIRED_TRAINING_FIELDS = frozenset(
    {
        "model_config",
        "tokenizer_config",
        "unit_read_config",
        "checkpoint_dir",
        "checkpoint_max_to_keep",
        "max_steps",
        "seed",
        "log_every",
        "save_every",
        "optimization",
        "runtime",
    }
)

_REQUIRED_OPTIMIZATION_FIELDS = frozenset(
    {
        "learning_rate",
        "warmup_steps",
        "decay_start_step",
        "weight_decay",
        "beta1",
        "beta2",
        "adam_eps",
        "clip_grad_norm",
        "min_lr_ratio",
    }
)

_REQUIRED_RUNTIME_FIELDS = frozenset(
    {
        "name",
        "device_index",
        "global_device",
        "compile_update",
    }
)


def build_training_config(values: Mapping[str, Any], *, label: str = "training") -> TrainingConfig:
    """Build TrainingConfig from a mapping."""
    reject_required_only_fields(values, required=_REQUIRED_TRAINING_FIELDS, label=label)

    optimization_values = expect_mapping(values["optimization"], label=f"{label}.optimization")
    reject_required_only_fields(
        optimization_values,
        required=_REQUIRED_OPTIMIZATION_FIELDS,
        label=f"{label}.optimization",
    )
    runtime_values = expect_mapping(values["runtime"], label=f"{label}.runtime")
    reject_required_only_fields(
        runtime_values,
        required=_REQUIRED_RUNTIME_FIELDS,
        label=f"{label}.runtime",
    )

    config = TrainingConfig(
        model_config=str(values["model_config"]),
        tokenizer_config=str(values["tokenizer_config"]),
        unit_read_config=str(values["unit_read_config"]),
        checkpoint_dir=str(values["checkpoint_dir"]),
        checkpoint_max_to_keep=int(values["checkpoint_max_to_keep"]),
        max_steps=int(values["max_steps"]),
        seed=int(values["seed"]),
        log_every=int(values["log_every"]),
        save_every=int(values["save_every"]),
        optimization=OptimizationConfig(
            learning_rate=float(optimization_values["learning_rate"]),
            warmup_steps=int(optimization_values["warmup_steps"]),
            decay_start_step=int(optimization_values["decay_start_step"]),
            weight_decay=float(optimization_values["weight_decay"]),
            beta1=float(optimization_values["beta1"]),
            beta2=float(optimization_values["beta2"]),
            adam_eps=float(optimization_values["adam_eps"]),
            clip_grad_norm=float(optimization_values["clip_grad_norm"]),
            min_lr_ratio=float(optimization_values["min_lr_ratio"]),
        ),
        runtime=RuntimeConfig(
            name=str(runtime_values["name"]),
            device_index=expect_plain_int(
                runtime_values["device_index"],
                label=f"{label}.runtime.device_index",
            ),
            global_device=expect_bool(
                runtime_values["global_device"],
                label=f"{label}.runtime.global_device",
            ),
            compile_update=expect_bool(
                runtime_values["compile_update"],
                label=f"{label}.runtime.compile_update",
            ),
        ),
    )
    config.validate()
    return config


def load_training_config_yaml(path: str | Path) -> TrainingConfig:
    """Load TrainingConfig from YAML."""
    config_path = Path(path)
    return build_training_config(
        read_yaml_mapping(config_path, purpose="training"),
        label=config_path.stem,
    )


__all__ = [
    "OptimizationConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "build_training_config",
    "load_training_config_yaml",
]
