"""CLI command for counting model parameters from model and tokenizer configs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from mintdim_lab.cli.program import command_prog
from mintdim_lab.model import Transformer, load_text_config_yaml
from mintdim_lab.system.paths import resolve_path_from_base
from mintdim_lab.tokenizer.config import load_tokenizer_config_yaml


def main(argv: list[str] | None = None) -> int:
    """Run the params command."""
    parser = build_parser()

    try:
        ns = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
        result = run_params_command(ns)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=command_prog("params"),
        description="Count actual Transformer parameters from model and tokenizer configs.",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--model-config",
        default="recipes/models/tiny.yaml",
        help="Model YAML path relative to repo unless absolute.",
    )
    parser.add_argument(
        "--tokenizer-config",
        default="recipes/tokenizers/byte_bpe_512.yaml",
        help="Tokenizer YAML path relative to repo unless absolute.",
    )
    return parser


def run_params_command(ns: argparse.Namespace) -> dict[str, Any]:
    repo = Path(ns.repo).resolve()
    model_config_path = resolve_path_from_base(repo, ns.model_config)
    tokenizer_config_path = resolve_path_from_base(repo, ns.tokenizer_config)

    with _working_directory(repo):
        tokenizer = load_tokenizer_config_yaml(tokenizer_config_path)
    model_config = load_text_config_yaml(
        model_config_path,
        num_embed=tokenizer.vocab_size,
    )
    variables = jax.eval_shape(
        lambda rng, tokens: Transformer(model_config).init(rng, tokens),
        jax.random.PRNGKey(0),
        jax.ShapeDtypeStruct((1, 1), jnp.int32),
    )
    params = int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(variables["params"])))

    return {
        "status": "ok",
        "repo": str(repo),
        "model_config": _repo_relative_string(repo, model_config_path),
        "tokenizer_config": _repo_relative_string(repo, tokenizer_config_path),
        "tokenizer": {
            "type": tokenizer.type,
            "path": tokenizer.path,
            "vocab_size": tokenizer.vocab_size,
        },
        "model": {
            "params": params,
            "params_m": round(params / 1_000_000, 6),
            "num_embed": model_config.num_embed,
            "num_layers": model_config.num_layers,
            "embed_dim": model_config.embed_dim,
            "dense_hidden_dim": model_config.dense_hidden_dim,
            "ffw_activation": model_config.ffw_activation,
            "ffn_ratio": model_config.dense_hidden_dim / model_config.embed_dim,
        },
    }


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _repo_relative_string(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return str(path.resolve())


__all__ = ["build_parser", "main", "run_params_command"]


if __name__ == "__main__":
    raise SystemExit(main())
