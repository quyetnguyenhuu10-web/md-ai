"""Command-line entry for the chat worker.

`worker/__init__.py` imports `protocol` first so the module-level stdout
redirect runs before any inference imports load JAX/Flax (which may emit
notices).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from mintdim_lab.inference.model_loader import load_runtime
from mintdim_lab.serving.chat_session import prompt_loop, warm_up
from mintdim_lab.serving.worker_protocol import write_error, write_event
from mintdim_lab.system.env import force_utf8_stdio
from mintdim_lab.system.paths import CHAT_CHECKPOINT_LIST_PATH, resolve_repo_path


def default_checkpoint() -> tuple[Path | None, Path | None]:
    """Return the first (checkpoint, vocab) pair that resolves on disk."""
    if not CHAT_CHECKPOINT_LIST_PATH.exists():
        return None, None

    data = yaml.safe_load(CHAT_CHECKPOINT_LIST_PATH.read_text(encoding="utf-8")) or {}
    for value in data.values():
        raw_paths = value.get("checkpoint", [])
        if not raw_paths or raw_paths[0] is None:
            continue

        ckpt = resolve_repo_path(raw_paths[0])
        if not ckpt.exists():
            continue

        raw_vocabs = value.get("vocab", [])
        vocab = resolve_repo_path(raw_vocabs[0]) if raw_vocabs and raw_vocabs[0] else None
        return ckpt, vocab

    return None, None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MintDim chat worker (stdio).")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint_dir = None
    args.preset = None
    args.no_print_decoded_prompt = True
    return args


def resolve_checkpoint(args: argparse.Namespace) -> None:
    if args.checkpoint is not None:
        args.checkpoint = resolve_repo_path(args.checkpoint)
    if args.vocab_path is not None:
        args.vocab_path = resolve_repo_path(args.vocab_path)
    if args.checkpoint is None or args.vocab_path is None:
        default_ckpt, default_vocab = default_checkpoint()
        args.checkpoint = args.checkpoint or default_ckpt
        args.vocab_path = args.vocab_path or default_vocab
    if args.checkpoint is None:
        write_error("No checkpoint available. Pass --checkpoint or fill checkpoints.yaml.")
        sys.exit(2)
    if args.vocab_path is None:
        write_error("No vocab path available. Pass --vocab-path or fill checkpoints.yaml.")
        sys.exit(2)
    if not args.checkpoint.exists():
        write_error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(2)
    if not args.vocab_path.exists():
        write_error(f"Vocab not found: {args.vocab_path}")
        sys.exit(2)


def main(argv: list[str] | None = None) -> None:
    force_utf8_stdio()
    args = parse_args(argv)
    resolve_checkpoint(args)

    write_event(
        {
            "event": "loading",
            "checkpoint": str(args.checkpoint),
            "vocab": str(args.vocab_path),
        }
    )

    model, variables, config, tokenizer, generate_fn, rng = load_runtime(
        args, vocab_path=args.vocab_path
    )

    if not args.no_warmup:
        write_event({"event": "warming-up"})
        rng = warm_up(
            generate_fn=generate_fn,
            model=model,
            variables=variables,
            config=config,
            tokenizer=tokenizer,
            rng=rng,
            raw_prompt=bool(args.raw_prompt),
        )

    write_event({"event": "ready"})

    prompt_loop(
        args=args,
        model=model,
        variables=variables,
        config=config,
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        rng=rng,
    )


__all__ = ["default_checkpoint", "main", "parse_args", "resolve_checkpoint"]
