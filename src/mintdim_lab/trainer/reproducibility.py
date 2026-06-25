"""Optional deterministic-training debug helpers.

This module is intentionally passive: it hashes already-built training objects
and writes JSONL records only when the train CLI explicitly enables it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import jax
import numpy as np

from mintdim_lab.system.paths import file_sha256


def tree_sha256(value: Any) -> str:
    """Return a deterministic SHA256 for a JAX/Python pytree."""
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest = hashlib.sha256()
    digest.update(str(treedef).encode("utf-8"))

    for leaf in leaves:
        arr = np.asarray(jax.device_get(leaf))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(arr).tobytes())

    return digest.hexdigest()


def ordered_hash_sha256(items: Sequence[str]) -> str:
    """Hash a sequence of precomputed hashes while preserving order."""
    digest = hashlib.sha256()
    digest.update(str(len(items)).encode("utf-8"))
    for item in items:
        digest.update(str(item).encode("ascii"))
    return digest.hexdigest()


def state_hashes(*, params: Any, opt_state: Any) -> dict[str, str]:
    """Hash train-state fields that determine future updates."""
    return {
        "params_sha256": tree_sha256(params),
        "opt_state_sha256": tree_sha256(opt_state),
    }


def runtime_fingerprint(
    *,
    runtime_name: str,
    device: Any,
    compile_update: bool,
    global_device: bool,
) -> dict[str, Any]:
    """Return stable runtime metadata useful for comparing two runs."""
    try:
        import jaxlib

        jaxlib_version = getattr(jaxlib, "__version__", None)
    except Exception:
        jaxlib_version = None

    return {
        "runtime": str(runtime_name),
        "compile_update": bool(compile_update),
        "global_device": bool(global_device),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": getattr(jax, "__version__", None),
        "jaxlib": jaxlib_version,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_platform": getattr(device, "platform", None),
        "device_kind": getattr(device, "device_kind", None),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def unit_read_source_fingerprints(
    *,
    entries: Sequence[Mapping[str, Any]],
    repo: Path,
) -> list[dict[str, Any]]:
    """Fingerprint files referenced by unit_read queue entries."""
    out: list[dict[str, Any]] = []
    for entry in entries:
        raw_path = Path(str(entry["path"]))
        path = raw_path if raw_path.is_absolute() else repo / raw_path
        item: dict[str, Any] = {
            "path": str(entry["path"]),
            "resolved_path": str(path),
            "unit": int(entry["unit"]),
            "batch": int(entry["batch"]),
            "accum": int(entry["accum"]),
            "exists": path.exists(),
        }
        if path.is_file():
            item["files"] = [_file_fingerprint(path, base=path.parent)]
        elif path.is_dir():
            files = sorted(p for p in path.iterdir() if p.is_file())
            item["files"] = [_file_fingerprint(p, base=path) for p in files]
        else:
            item["files"] = []
        out.append(item)
    return out


def write_jsonl(handle: TextIO, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
    handle.write("\n")
    handle.flush()


def _file_fingerprint(path: Path, *, base: Path) -> dict[str, Any]:
    return {
        "name": path.relative_to(base).as_posix(),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


__all__ = [
    "ordered_hash_sha256",
    "runtime_fingerprint",
    "state_hashes",
    "tree_sha256",
    "unit_read_source_fingerprints",
    "write_jsonl",
]
