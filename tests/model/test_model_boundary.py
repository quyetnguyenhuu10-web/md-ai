from __future__ import annotations

import importlib
from pathlib import Path

MODEL_MODULES = (
    "attention",
    "block",
    "builder",
    "config",
    "embedding",
    "rope",
    "transformer",
)

MODEL_LAYER_MODULES = (
    "activation",
    "linear",
    "norm",
)


def _module_path(module) -> str:
    raw = getattr(module, "__file__", None)
    assert raw is not None
    return Path(raw).resolve().as_posix()


def test_model_modules_resolve_to_production_boundary():
    for name in MODEL_MODULES:
        module = importlib.import_module(f"mintdim_lab.model.{name}")

        assert module.__name__ == f"mintdim_lab.model.{name}"
        assert "/src/mintdim_lab/model/" in _module_path(module)


def test_model_layer_modules_resolve_to_production_boundary():
    for name in MODEL_LAYER_MODULES:
        module = importlib.import_module(f"mintdim_lab.model.layers.{name}")

        assert module.__name__ == f"mintdim_lab.model.layers.{name}"
        assert "/src/mintdim_lab/model/layers/" in _module_path(module)


def test_model_packages_are_the_production_namespaces():
    model_package = importlib.import_module("mintdim_lab.model")
    layers_package = importlib.import_module("mintdim_lab.model.layers")

    assert model_package.__name__ == "mintdim_lab.model"
    assert layers_package.__name__ == "mintdim_lab.model.layers"
    assert "/src/mintdim_lab/model/" in _module_path(model_package)
    assert "/src/mintdim_lab/model/layers/" in _module_path(layers_package)


def test_model_public_boundary_exports_transformer_entrypoint():
    from mintdim_lab.model import Transformer
    from mintdim_lab.model.transformer import Transformer as TransformerImpl

    assert Transformer is TransformerImpl
