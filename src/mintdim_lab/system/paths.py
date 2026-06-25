from __future__ import annotations

import hashlib
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for current in (start, *start.parents):
        if (current / "pyproject.toml").is_file() and (current / "src").is_dir():
            return current
        if (current / "behavior_locks").is_dir() and (current / "src").is_dir():
            return current

    return Path(__file__).resolve().parents[3]


SYSTEM_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = _find_repo_root(Path(__file__).resolve())
PACKAGE_ROOT = SYSTEM_PACKAGE_ROOT

CONFIG_DIR = REPO_ROOT / "configs"
DATA_STORE_ROOT = REPO_ROOT / "data_store"

DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "runs" / "checkpoints"
TOKENIZER_DIR = DATA_STORE_ROOT / "tokenizers" / "byte_bpe_512"
DEFAULT_VOCAB_PATH = TOKENIZER_DIR / "tokenizer.json"
CHAT_CHECKPOINT_LIST_PATH = REPO_ROOT / "recipes" / "chat" / "checkpoints.yaml"


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_path_from_base(base: Path, path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
