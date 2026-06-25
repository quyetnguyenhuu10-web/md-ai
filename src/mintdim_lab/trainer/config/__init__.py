"""Public trainer config surface.

Tokenizer metadata lives in ``mintdim_lab.tokenizer.config`` and is re-exported
here because training setup needs vocab metadata before model initialization.

Implementation is split by responsibility under ``mintdim_lab.trainer.config``:
- ``training`` owns run/runtime/optimizer config.
- ``unit_read`` owns public MintDim pipeline("unit-read") queue/layout/target.
"""

from __future__ import annotations

from mintdim_lab.corpus.config_paths import resolve_unit_read_config_path
from mintdim_lab.tokenizer.config import (
    TokenizerConfig,
    build_tokenizer_config,
    load_tokenizer_config_yaml,
    read_hf_json_tokenizer_metadata,
)
from mintdim_lab.trainer.config.training import (
    OptimizationConfig,
    RuntimeConfig,
    TrainingConfig,
    build_training_config,
    load_training_config_yaml,
)
from mintdim_lab.trainer.config.unit_read import (
    UnitReadConfig,
    UnitReadLayoutConfig,
    UnitReadQueueEntry,
    UnitReadTargetConfig,
    build_unit_read_config,
    load_unit_read_config_yaml,
    validate_units_within_max_seq_len,
)

__all__ = [
    "OptimizationConfig",
    "RuntimeConfig",
    "TokenizerConfig",
    "TrainingConfig",
    "UnitReadConfig",
    "UnitReadLayoutConfig",
    "UnitReadQueueEntry",
    "UnitReadTargetConfig",
    "build_tokenizer_config",
    "build_training_config",
    "build_unit_read_config",
    "load_tokenizer_config_yaml",
    "load_training_config_yaml",
    "load_unit_read_config_yaml",
    "read_hf_json_tokenizer_metadata",
    "resolve_unit_read_config_path",
    "validate_units_within_max_seq_len",
]
