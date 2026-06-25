"""Orbax checkpoint helpers for training state.

This module owns checkpoint persistence policy for mintdim_lab.trainer. Runtime adapters
remain checkpoint-agnostic.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def should_save_checkpoint(*, step: int, save_every: int) -> bool:
    """Return whether a completed optimizer step should be checkpointed."""
    if int(save_every) <= 0:
        raise ValueError("save_every must be positive")
    return int(step) > 0 and int(step) % int(save_every) == 0


def checkpoint_step_dir(checkpoint_dir: str | Path, *, step: int) -> Path:
    """Return the deterministic directory for one checkpoint step."""
    if int(step) <= 0:
        raise ValueError("checkpoint step must be positive")
    return Path(checkpoint_dir) / f"step_{int(step):08d}"


def save_train_state_checkpoint(
    *,
    checkpoint_dir: str | Path,
    state: Any,
    step: int,
    max_to_keep: int,
    metadata: Mapping[str, Any] | None = None,
    config_files: Mapping[str, str | Path] | None = None,
) -> Path:
    """Save TrainState payload plus portable rebuild metadata.

    The Orbax payload stores:
    - params
    - opt_state
    - step
    - metadata

    The checkpoint directory also gets sidecar files:
    - metadata.json
    - copied config files under their repo-relative paths

    This keeps inference/benchmark checkpoint loading deterministic: it can
    rebuild the model from checkpoint metadata without guessing config files.
    """
    if int(step) <= 0:
        raise ValueError("step must be positive")

    resolved_step = int(step)
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)

    target = checkpoint_step_dir(root, step=resolved_step)
    if target.exists():
        shutil.rmtree(target)

    resolved_metadata = dict(metadata or {})
    resolved_metadata.setdefault("step", resolved_step)
    resolved_metadata.setdefault("array_storage", "host_numpy")
    resolved_metadata.setdefault("runtime_topology_dependent", False)

    # Keep the Orbax payload tensor/state-only. Full metadata contains strings,
    # YAML mappings, hashes, and paths, which StandardCheckpointer does not
    # accept as PyTree leaves.
    payload = {
        "params": portable_array_tree(state.params),
        "opt_state": portable_array_tree(state.opt_state),
        "step": resolved_step,
    }

    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(target, payload)
        wait = getattr(checkpointer, "wait_until_finished", None)
        if wait is not None:
            wait()
    finally:
        close = getattr(checkpointer, "close", None)
        if close is not None:
            close()

    write_checkpoint_sidecars(
        target,
        metadata=resolved_metadata,
        config_files=config_files,
    )

    prune_old_checkpoints(checkpoint_dir=root, max_to_keep=int(max_to_keep))
    return target


def portable_array_tree(value: Any) -> Any:
    """Return a checkpoint payload tree detached from active device topology.

    Training may run on CPU, GPU, or TPU, but saved checkpoints should not
    require the same runtime topology to restore. Converting array leaves to
    host NumPy arrays before Orbax sees them prevents device sharding metadata
    from becoming part of the saved payload.
    """
    import jax

    def convert_leaf(leaf: Any) -> Any:
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            return np.asarray(jax.device_get(leaf))
        return leaf

    return jax.tree_util.tree_map(convert_leaf, value)


def write_checkpoint_sidecars(
    checkpoint_path: Path,
    *,
    metadata: Mapping[str, Any],
    config_files: Mapping[str, str | Path] | None,
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    metadata_path = checkpoint_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(json_safe(dict(metadata)), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not config_files:
        return

    for raw_dest, raw_source in config_files.items():
        dest_rel = Path(str(raw_dest))
        if dest_rel.is_absolute() or ".." in dest_rel.parts:
            raise ValueError(
                f"checkpoint config destination must be relative and safe: {raw_dest!r}"
            )

        source = Path(raw_source)
        if not source.exists():
            raise FileNotFoundError(f"checkpoint config source not found: {source}")

        dest = checkpoint_path / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def list_checkpoint_steps(checkpoint_dir: str | Path) -> tuple[int, ...]:
    """List saved checkpoint step numbers."""
    root = Path(checkpoint_dir)
    if not root.exists():
        return ()

    steps: list[int] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        if not item.name.startswith("step_"):
            continue
        suffix = item.name.removeprefix("step_")
        if not suffix.isdigit():
            continue
        steps.append(int(suffix))

    return tuple(sorted(steps))


def latest_checkpoint_step(checkpoint_dir: str | Path) -> int | None:
    """Return the latest saved checkpoint step, if any."""
    steps = list_checkpoint_steps(checkpoint_dir)
    if not steps:
        return None
    return steps[-1]


def prune_old_checkpoints(*, checkpoint_dir: str | Path, max_to_keep: int) -> tuple[int, ...]:
    """Keep only the latest max_to_keep step checkpoint directories."""
    if int(max_to_keep) <= 0:
        raise ValueError("max_to_keep must be positive")

    root = Path(checkpoint_dir)
    steps = list_checkpoint_steps(root)
    to_delete = steps[: max(0, len(steps) - int(max_to_keep))]

    for step in to_delete:
        path = checkpoint_step_dir(root, step=step)
        if path.exists():
            shutil.rmtree(path)

    return tuple(to_delete)


_json_safe = json_safe
_portable_array_tree = portable_array_tree
_write_checkpoint_sidecars = write_checkpoint_sidecars


__all__ = [
    "checkpoint_step_dir",
    "json_safe",
    "latest_checkpoint_step",
    "list_checkpoint_steps",
    "portable_array_tree",
    "prune_old_checkpoints",
    "save_train_state_checkpoint",
    "should_save_checkpoint",
    "write_checkpoint_sidecars",
]
