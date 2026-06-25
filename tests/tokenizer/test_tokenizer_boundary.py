from __future__ import annotations

import importlib
from pathlib import Path

TOKENIZER_MODULES = (
    "audit",
    "config",
    "hf_json",
    "rules",
    "train_byte_bpe",
)


def _module_path(module) -> str:
    raw = getattr(module, "__file__", None)
    assert raw is not None
    return Path(raw).resolve().as_posix()


def test_tokenizer_modules_resolve_to_production_boundary():
    for name in TOKENIZER_MODULES:
        module = importlib.import_module(f"mintdim_lab.tokenizer.{name}")

        assert module.__name__ == f"mintdim_lab.tokenizer.{name}"
        assert "/src/mintdim_lab/tokenizer/" in _module_path(module)


def test_tokenizer_package_is_the_production_namespace():
    package = importlib.import_module("mintdim_lab.tokenizer")

    assert package.__name__ == "mintdim_lab.tokenizer"
    assert "/src/mintdim_lab/tokenizer/" in _module_path(package)


def test_tokenizer_public_boundary_imports_config_entrypoints():
    hf_json = importlib.import_module("mintdim_lab.tokenizer.hf_json")
    rules = importlib.import_module("mintdim_lab.tokenizer.rules")
    train_byte_bpe = importlib.import_module("mintdim_lab.tokenizer.train_byte_bpe")

    from mintdim_lab.tokenizer.config import load_tokenizer_config_yaml
    from mintdim_lab.trainer.config import load_training_config_yaml

    assert callable(load_tokenizer_config_yaml)
    assert callable(load_training_config_yaml)
    assert hf_json.__name__ == "mintdim_lab.tokenizer.hf_json"
    assert rules.__name__ == "mintdim_lab.tokenizer.rules"
    assert train_byte_bpe.__name__ == "mintdim_lab.tokenizer.train_byte_bpe"
