from __future__ import annotations

import importlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SERVING_MODULES = (
    "chat_session",
    "http_api",
    "prompting",
    "worker_cli",
    "worker_protocol",
)


def _module_path(module) -> str:
    raw = getattr(module, "__file__", None)
    assert raw is not None
    return Path(raw).resolve().as_posix()


def test_serving_modules_resolve_to_production_boundary():
    for name in SERVING_MODULES:
        module = importlib.import_module(f"mintdim_lab.serving.{name}")

        assert module.__name__ == f"mintdim_lab.serving.{name}"
        assert "/src/mintdim_lab/serving/" in _module_path(module)


def test_serving_package_is_the_production_namespace():
    package = importlib.import_module("mintdim_lab.serving")

    assert package.__name__ == "mintdim_lab.serving"
    assert "/src/mintdim_lab/serving/" in _module_path(package)


def test_serving_public_boundary_imports_worker_and_generation_entrypoints():
    from mintdim_lab.serving.prompting import chat_prompt_for_model
    from mintdim_lab.serving.worker_cli import main as worker_main
    from mintdim_lab.serving.worker_cli import parse_args
    from mintdim_lab.serving.worker_protocol import write_error, write_event

    assert callable(chat_prompt_for_model)
    assert callable(worker_main)
    assert callable(parse_args)
    assert callable(write_error)
    assert callable(write_event)


def test_worker_cli_resolves_relative_paths_from_repo_root(monkeypatch):
    from mintdim_lab.serving.worker_cli import parse_args, resolve_checkpoint

    monkeypatch.chdir(REPO / "apps" / "terminal")
    args = parse_args(
        [
            "--checkpoint",
            "recipes/chat/checkpoints.yaml",
            "--vocab-path",
            "data_store/tokenizers/byte_bpe_512/tokenizer.json",
        ]
    )

    resolve_checkpoint(args)

    assert args.checkpoint == (REPO / "recipes" / "chat" / "checkpoints.yaml").resolve()
    assert args.vocab_path == (
        REPO / "data_store" / "tokenizers" / "byte_bpe_512" / "tokenizer.json"
    ).resolve()
