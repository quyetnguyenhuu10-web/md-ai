from __future__ import annotations

from pathlib import Path

from mintdim_lab.system.checkpoint_io import (
    CHECKPOINT_SUFFIXES,
    find_checkpoints,
    resolve_checkpoint_dir,
)
from mintdim_lab.system.paths import DEFAULT_CHECKPOINT_DIR


def select_checkpoint(checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        suffixes = ", ".join(sorted(CHECKPOINT_SUFFIXES))
        raise SystemExit(
            f"No checkpoint files or Orbax checkpoint dirs found in {checkpoint_dir}\n"
            f"Supported suffixes: {suffixes}"
        )

    print(f"Checkpoints in {checkpoint_dir}:")
    print(f"Supported suffixes: {', '.join(sorted(CHECKPOINT_SUFFIXES))}; Orbax dirs")
    for index, path in enumerate(checkpoints, start=1):
        display_path = path.relative_to(checkpoint_dir)
        if path.is_dir():
            print(f"{index:>2}. {display_path}  [orbax dir]")
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{index:>2}. {display_path}  [{path.suffix}, {size_mb:.1f} MB]")

    while True:
        raw = input("Select checkpoint number or path: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(checkpoints):
            return checkpoints[int(raw) - 1]

        raw_path = Path(raw)
        candidate = raw_path if raw_path.is_absolute() else checkpoint_dir / raw_path
        if candidate in checkpoints:
            return candidate.resolve()
        if candidate.is_file() and candidate.suffix.lower() in CHECKPOINT_SUFFIXES:
            return candidate.resolve()
        if candidate.is_dir() and (
            candidate.name.startswith("step_") or (candidate / "metadata.json").is_file()
        ):
            return candidate.resolve()

        print("Invalid selection.")
