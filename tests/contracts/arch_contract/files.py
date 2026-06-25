from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from .config import ContractConfig

SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".mdx"}
PYTHON_SUFFIXES = {".py"}


def relpath(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def path_matches(path: str, patterns: list[str]) -> bool:
    posix = PurePosixPath(path)
    return any(posix.match(pattern) or fnmatch(path, pattern) for pattern in patterns)


def is_ignored(path: Path, config: ContractConfig) -> bool:
    relative = relpath(path, config.repo_root)
    parts = set(PurePosixPath(relative).parts)
    for ignored in config.ignored_dirs:
        if "/" not in ignored and ignored in parts:
            return True
        if path_matches(relative, [ignored, f"{ignored}/**", f"**/{ignored}/**"]):
            return True
    return False


def iter_source_files(config: ContractConfig, suffixes: set[str] = SOURCE_SUFFIXES) -> list[Path]:
    files: list[Path] = []
    for root in config.source_roots:
        absolute_root = config.repo_root / root
        if not absolute_root.exists():
            continue
        if absolute_root.is_file():
            candidates = [absolute_root]
        else:
            candidates = [path for path in absolute_root.rglob("*") if path.is_file()]
        for path in candidates:
            if path.suffix in suffixes and not is_ignored(path, config):
                files.append(path)
    return sorted(set(files))


def iter_python_files(config: ContractConfig) -> list[Path]:
    return iter_source_files(config, PYTHON_SUFFIXES)


def find_layer_for_path(path: Path, config: ContractConfig) -> str | None:
    relative = relpath(path, config.repo_root)
    for layer_name, layer in config.layers.items():
        patterns = list(layer.get("patterns", []))
        if path_matches(relative, patterns):
            return layer_name
    return None


def is_composition_root(path: Path, config: ContractConfig) -> bool:
    relative = relpath(path, config.repo_root)
    return path_matches(relative, list(config.data.get("composition_roots", [])))
