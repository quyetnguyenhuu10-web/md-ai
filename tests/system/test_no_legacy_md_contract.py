from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY_PACKAGE = "m" + "d"

TEXT_SUFFIXES = {
    "",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

PYTHON_IMPORT_ROOTS = (
    "src",
    "tests",
    "behavior_locks",
)

PRODUCTION_TEXT_ROOTS = (
    "src",
    "apps",
)

PUBLIC_DOC_ROOTS = (
    "README.md",
    "apps/terminal/package.json",
    "behavior_locks",
    "docs",
)

HISTORICAL_DOC_ALLOWLIST = {
    "docs/plan_for_agent/plan_patch.md",
    "docs/refactor_domain_driven_phase_plan.md",
}


def test_legacy_md_package_is_absent() -> None:
    assert importlib.util.find_spec(LEGACY_PACKAGE) is None
    assert not (REPO / LEGACY_PACKAGE).exists()


def test_no_python_sources_import_legacy_md() -> None:
    forbidden = (
        f"from {LEGACY_PACKAGE}.",
        f"import {LEGACY_PACKAGE}.",
    )
    hits = _find_lines(PYTHON_IMPORT_ROOTS, forbidden, suffixes={".py"})

    assert hits == []


def test_production_sources_contain_no_legacy_md_examples() -> None:
    forbidden = (
        f"from {LEGACY_PACKAGE}.",
        f"import {LEGACY_PACKAGE}.",
        f"python -m {LEGACY_PACKAGE}.",
        f"{LEGACY_PACKAGE}/chat",
        f"{LEGACY_PACKAGE}/vocab",
        f"{LEGACY_PACKAGE}/data",
        f"{LEGACY_PACKAGE}/nn",
        f"{LEGACY_PACKAGE}/training",
        f"{LEGACY_PACKAGE}/benchmark",
        f"{LEGACY_PACKAGE}/foundation",
        f"{LEGACY_PACKAGE}/runtime",
        f"{LEGACY_PACKAGE}.train_state.orbax",
    )
    hits = _find_lines(PRODUCTION_TEXT_ROOTS, forbidden)

    assert hits == []


def test_public_docs_do_not_instruct_legacy_md_usage() -> None:
    forbidden = (
        f"python -m {LEGACY_PACKAGE}.",
        f"bun {LEGACY_PACKAGE}/",
        f"{LEGACY_PACKAGE}/chat",
        f"{LEGACY_PACKAGE}/vocab",
        f"{LEGACY_PACKAGE}/data",
        f"{LEGACY_PACKAGE}/nn",
        f"{LEGACY_PACKAGE}/training",
        f"{LEGACY_PACKAGE}/benchmark",
        f"{LEGACY_PACKAGE}/foundation",
        f"{LEGACY_PACKAGE}/runtime",
        f"`{LEGACY_PACKAGE}.",
        f'"{LEGACY_PACKAGE}.',
        f"'{LEGACY_PACKAGE}.",
    )
    hits = _find_lines(
        PUBLIC_DOC_ROOTS,
        forbidden,
        excluded_relative_paths=HISTORICAL_DOC_ALLOWLIST,
    )

    assert hits == []


def _find_lines(
    roots: tuple[str, ...],
    forbidden: tuple[str, ...],
    *,
    suffixes: set[str] = TEXT_SUFFIXES,
    excluded_relative_paths: set[str] | None = None,
) -> list[str]:
    excluded = excluded_relative_paths or set()
    hits: list[str] = []

    for path in _iter_text_paths(roots, suffixes=suffixes):
        relative = path.relative_to(REPO).as_posix()
        if relative in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(fragment in line for fragment in forbidden):
                hits.append(f"{relative}:{line_no}: {line}")

    return hits


def _iter_text_paths(roots: tuple[str, ...], *, suffixes: set[str]) -> list[Path]:
    paths: list[Path] = []

    for root_name in roots:
        root = REPO / root_name
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if path.suffix.lower() not in suffixes:
                continue
            paths.append(path)

    return paths
