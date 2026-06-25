"""CLI launcher for the MintDim chat UI and stdio worker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from mintdim_lab.cli.program import command_prog


def main(argv: list[str] | None = None) -> int:
    """Run the chat command."""
    parser = build_parser()
    ns, passthrough = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))

    if bool(ns.worker):
        return _run_worker(passthrough)

    return _run_ui(
        repo=Path(ns.repo).resolve(),
        bun=str(ns.bun),
        python=None if ns.python is None else str(ns.python),
        passthrough=passthrough,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog("chat"),
        description="Launch the MintDim chat TUI or its JSON stdio worker.",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run the Python JSON stdio worker instead of the Bun TUI.",
    )
    parser.add_argument(
        "--bun",
        default="bun",
        help="Bun executable used for the TUI. Default: bun.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable passed to the TUI worker launcher, e.g. python or python3.",
    )
    return parser


def _run_ui(
    *,
    repo: Path,
    bun: str,
    python: str | None,
    passthrough: list[str],
) -> int:
    app_dir = repo / "apps" / "terminal"
    script = app_dir / "chat.tsx"
    if not script.is_file():
        print(f"FileNotFoundError: chat UI entrypoint not found: {script}", file=sys.stderr)
        return 1

    cmd = [bun, "chat.tsx"]
    if python is not None:
        cmd.extend(["--python", python])
    cmd.extend(passthrough)

    try:
        return int(subprocess.call(cmd, cwd=app_dir))
    except FileNotFoundError:
        print(f"FileNotFoundError: Bun executable not found: {bun}", file=sys.stderr)
        return 1


def _run_worker(worker_args: list[str]) -> int:
    from mintdim_lab.serving.worker_cli import main as worker_main

    try:
        worker_main(worker_args)
    except SystemExit as exc:
        return _exit_code(exc.code)
    return 0


def _exit_code(code: Any) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


__all__ = ["build_parser", "main"]
