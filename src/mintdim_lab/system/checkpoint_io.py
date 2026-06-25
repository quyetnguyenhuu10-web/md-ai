from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mintdim_lab.system.paths import resolve_repo_path

CHECKPOINT_SUFFIXES = {
    ".msgpack",
    ".npz",
    ".pkl",
    ".pickle",
    ".ckpt",
    ".safetensors",
    ".pt",
    ".pth",
}


def resolve_checkpoint_path(path: Path | str) -> Path:
    return resolve_repo_path(path)


def resolve_checkpoint_dir(path: Path | str) -> Path:
    return resolve_checkpoint_path(path)


def find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    return sorted(
        (path for path in checkpoint_dir.rglob("*") if _is_supported_checkpoint_path(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_checkpoint(path: Path) -> Any:
    path = resolve_checkpoint_path(path)
    if _is_orbax_training_checkpoint(path):
        return _load_orbax_params_checkpoint(path)
    suffix = path.suffix.lower()
    if suffix == ".msgpack":
        from flax import serialization

        return serialization.msgpack_restore(path.read_bytes())
    if suffix == ".npz":
        return _load_npz(path)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if suffix == ".ckpt":
        return _load_ckpt(path)
    if suffix == ".safetensors":
        from safetensors.flax import load_file

        return load_file(str(path))
    if suffix in {".pt", ".pth"}:
        return _load_torch_checkpoint(path)
    raise ValueError(f"Unsupported checkpoint suffix: {suffix}")


def _is_supported_checkpoint_path(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in CHECKPOINT_SUFFIXES
    return _is_orbax_training_checkpoint(path)


def checkpoint_to_variables(checkpoint: Any) -> dict[str, Any]:
    if hasattr(checkpoint, "params"):
        checkpoint = checkpoint.params
    if isinstance(checkpoint, Mapping):
        value = checkpoint.get("params")
        if isinstance(value, Mapping):
            return {"params": value}
        for key in ("params", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
        if "params" in checkpoint:
            return checkpoint
    return {"params": checkpoint}


def checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, Mapping):
        metadata = checkpoint.get("metadata")
        if isinstance(metadata, Mapping):
            return dict(metadata)
    return {}


def checkpoint_preset(checkpoint: Any, *, default: str = "dense_5m") -> str:
    metadata = checkpoint_metadata(checkpoint)
    return str(metadata.get("preset") or default)


def checkpoint_logits_head(
    checkpoint: Any,
    *,
    variables: Mapping[str, Any] | None = None,
    default: str = "tied",
) -> str:
    metadata = checkpoint_metadata(checkpoint)
    raw = metadata.get("logits_head")
    if raw is not None:
        return str(raw).strip().lower()
    if variables is not None and _has_param_named(variables, "output_head"):
        return "untied"
    return default


def checkpoint_num_embed(
    checkpoint: Any,
    *,
    variables: Mapping[str, Any] | None = None,
) -> int | None:
    metadata = checkpoint_metadata(checkpoint)
    value = metadata.get("num_embed")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    if variables is None:
        return None
    return _find_input_embedding_size(variables)


def _find_input_embedding_size(value: Any) -> int | None:
    if isinstance(value, Mapping):
        embedding = value.get("input_embedding")
        if hasattr(embedding, "shape") and len(embedding.shape) >= 1:
            return int(embedding.shape[0])
        for item in value.values():
            found = _find_input_embedding_size(item)
            if found is not None:
                return found
    return None


def _has_param_named(value: Any, name: str) -> bool:
    if isinstance(value, Mapping):
        if name in value:
            return True
        return any(_has_param_named(item, name) for item in value.values())
    return False


def _load_npz(path: Path) -> dict[str, Any]:
    from flax.traverse_util import unflatten_dict

    data = np.load(path, allow_pickle=False)
    flat = {tuple(key.split("/")): data[key] for key in data.files}
    return unflatten_dict(flat)


def _load_ckpt(path: Path) -> Any:
    errors: list[str] = []

    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        errors.append(f"pickle: {exc}")

    try:
        from flax import serialization

        return serialization.msgpack_restore(path.read_bytes())
    except Exception as exc:
        errors.append(f"msgpack: {exc}")

    raise ValueError(f"Could not load .ckpt by suffix. Tried {'; '.join(errors)}")


def _load_torch_checkpoint(path: Path) -> Any:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    return _to_numpy(checkpoint)


def _read_orbax_sidecar_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return {}

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"checkpoint metadata sidecar must be a mapping: {metadata_path}")
    return dict(data)


def _to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_numpy(item) for item in value)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


# =============================================================================
# Repo-local Orbax TrainState checkpoint support
# =============================================================================


def _latest_orbax_step_dir(path: Path) -> Path | None:
    if not path.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("step_"):
            continue
        suffix = child.name.removeprefix("step_")
        if not suffix.isdigit():
            continue
        candidates.append((int(suffix), child))

    if not candidates:
        return None

    return sorted(candidates)[-1][1]


def _is_orbax_training_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith("step_"):
        return True
    return _latest_orbax_step_dir(path) is not None


def _load_orbax_params_checkpoint(path: Path) -> dict[str, Any]:
    resolved = _latest_orbax_step_dir(path) if not path.name.startswith("step_") else path
    if resolved is None:
        resolved = path

    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    try:
        payload = checkpointer.restore(resolved)
    finally:
        close = getattr(checkpointer, "close", None)
        if close is not None:
            close()

    if not isinstance(payload, Mapping):
        return {
            "params": payload,
            "metadata": {
                "checkpoint_path": str(resolved),
            },
        }

    metadata = _read_orbax_sidecar_metadata(resolved)
    payload_metadata = payload.get("metadata")
    if isinstance(payload_metadata, Mapping):
        metadata.update(payload_metadata)
    metadata.setdefault("checkpoint_path", str(resolved))

    step = payload.get("step")
    if step is not None:
        try:
            metadata.setdefault("step", int(step))
        except Exception:
            metadata.setdefault("step", step)

    params = payload.get("params")
    if params is None and "state" in payload:
        state = payload["state"]
        if isinstance(state, Mapping):
            params = state.get("params")

    result = dict(payload)
    if params is not None:
        result["params"] = params
    result["metadata"] = metadata
    return result
