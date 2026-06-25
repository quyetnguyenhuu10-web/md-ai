from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_train_metadata_translates_attention_types_to_attention_pattern():
    body = text("src/mintdim_lab/run_setup/checkpoint_metadata.py")

    assert '"attention_pattern" not in raw' in body
    assert 'hasattr(config, "attention_types")' in body
    assert 'raw["attention_pattern"]' in body
    assert 'getattr(item, "value", str(item))' in body


def test_checkpoint_metadata_module_exports_clean_contract():
    module = ast.parse(text("src/mintdim_lab/run_setup/checkpoint_metadata.py"))

    function_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}

    assert "model_config_metadata_values" in function_names
    assert "build_checkpoint_metadata" in function_names
