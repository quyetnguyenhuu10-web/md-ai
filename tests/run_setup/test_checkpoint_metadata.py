from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_train_checkpoint_store_accepts_metadata_and_config_files():
    body = text("src/mintdim_lab/system/checkpoint_store.py")

    assert "metadata: Mapping[str, Any] | None = None" in body
    assert "config_files: Mapping[str, str | Path] | None = None" in body
    assert '"metadata": resolved_metadata' not in body
    assert 'metadata_path = checkpoint_path / "metadata.json"' in body
    assert "StandardCheckpointer does not" in body
    assert "shutil.copy2(source, dest)" in body


def test_run_setup_checkpoint_metadata_builds_full_metadata_from_all_configs():
    body = text("src/mintdim_lab/run_setup/checkpoint_metadata.py")
    train_body = text("src/mintdim_lab/trainer/run_training.py")

    assert "def build_checkpoint_metadata(" in body
    assert '"training_config": repo_relative_string(repo, paths["training_config"])' in body
    assert '"tokenizer_config": repo_relative_string(repo, paths["tokenizer_config"])' in body
    assert '"unit_read_config": repo_relative_string(repo, paths["unit_read_config"])' in body
    assert '"model_config": repo_relative_string(repo, paths["model_config"])' in body
    assert '"configs": {' in body
    assert '"vocab_path": tokenizer_path' in body
    assert (
        '"model_config_metadata_fields": list(text_metadata_fields(include_num_embed=True))' in body
    )
    assert "metadata = checkpoint_metadata" in train_body
    assert "metadata_with_determinism_debug(" in train_body
    assert "metadata=metadata" in train_body
    assert "config_files=checkpoint_config_files" in train_body


def test_expected_config_files_exist_for_checkpoint_metadata():
    paths = [
        REPO / "recipes" / "train" / "cpu" / "linear_equation_tiny_cpu.yaml",
        REPO / "recipes" / "tokenizers" / "byte_bpe_512.yaml",
        REPO / "recipes" / "corpus" / "linear_equation_unit96" / "unit_read.yaml",
        REPO / TEST_MODEL_CONFIG,
    ]

    for path in paths:
        assert path.exists(), path


def test_metadata_sidecar_json_shape_example():
    example = {
        "metadata_version": 1,
        "config_paths": {
            "training_config": "recipes/train/cpu/linear_equation_tiny_cpu.yaml",
            "tokenizer_config": "recipes/tokenizers/byte_bpe_512.yaml",
            "unit_read_config": "recipes/corpus/linear_equation_unit96/unit_read.yaml",
            "model_config": TEST_MODEL_CONFIG,
        },
        "vocab_path": "./data_store/tokenizers/byte_bpe_512/tokenizer.json",
        "model_config_metadata_fields": ["num_embed"],
    }

    encoded = json.dumps(example, ensure_ascii=False)
    decoded = json.loads(encoded)

    assert (
        decoded["config_paths"]["training_config"]
        == "recipes/train/cpu/linear_equation_tiny_cpu.yaml"
    )
    assert decoded["vocab_path"] == "./data_store/tokenizers/byte_bpe_512/tokenizer.json"


def test_foundation_checkpoint_reads_metadata_sidecar():
    body = text("src/mintdim_lab/system/checkpoint_io.py")

    assert "def _read_orbax_sidecar_metadata(" in body
    assert 'metadata_path = path / "metadata.json"' in body
    assert "metadata = _read_orbax_sidecar_metadata(resolved)" in body
