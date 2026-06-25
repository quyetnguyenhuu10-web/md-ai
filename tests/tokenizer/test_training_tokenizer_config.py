from __future__ import annotations

import json
from pathlib import Path

from mintdim_lab.model import load_text_config_yaml
from mintdim_lab.tokenizer.config import (
    build_tokenizer_config,
    load_tokenizer_config_yaml,
    read_hf_json_tokenizer_metadata,
)

TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def test_read_hf_json_tokenizer_metadata_from_repo_vocab():
    path = (
        Path(__file__).resolve().parents[2]
        / "data_store"
        / "tokenizers"
        / "byte_bpe_512"
        / "tokenizer.json"
    )

    metadata = read_hf_json_tokenizer_metadata(path)

    assert metadata["vocab_size"] > 0
    assert metadata["pad_id"] == 0
    assert metadata["unk_id"] == 2
    assert metadata["eos_id"] == 1


def test_load_tokenizer_config_yaml_provides_vocab_size_for_model_config():
    repo = Path(__file__).resolve().parents[2]

    tokenizer_cfg = load_tokenizer_config_yaml(
        repo / "recipes" / "tokenizers" / "byte_bpe_512.yaml"
    )
    model_cfg = load_text_config_yaml(
        repo / TEST_MODEL_CONFIG,
        num_embed=tokenizer_cfg.vocab_size,
    )

    assert tokenizer_cfg.type == "hf_json"
    assert tokenizer_cfg.path == "./data_store/tokenizers/byte_bpe_512/tokenizer.json"
    assert model_cfg.num_embed == tokenizer_cfg.vocab_size


def test_tokenizer_config_does_not_belong_to_unit_read_yaml():
    import yaml

    unit_read_path = (
        Path(__file__).resolve().parents[2]
        / "recipes"
        / "corpus"
        / "linear_equation_unit96"
        / "unit_read.yaml"
    )
    unit_read_values = yaml.safe_load(unit_read_path.read_text(encoding="utf-8"))

    assert "tokenizer" not in unit_read_values
    assert set(unit_read_values) == {"queue", "layout", "target"}


def test_tokenizer_config_rejects_unknown_fields():
    values = {
        "type": "hf_json",
        "path": "./data_store/tokenizers/byte_bpe_512/tokenizer.json",
        "unexpected": True,
    }

    try:
        build_tokenizer_config(values)
    except ValueError as exc:
        assert "unknown fields" in str(exc)
    else:
        raise AssertionError("unknown tokenizer config fields should be rejected")


def test_tokenizer_metadata_uses_max_id_plus_one(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(
            {
                "padding": {"pad_id": 0, "pad_token": "<pad>"},
                "added_tokens": [
                    {"id": 0, "content": "<pad>"},
                    {"id": 1, "content": "<eos>"},
                    {"id": 2, "content": "<unk>"},
                    {"id": 9, "content": "<extra>"},
                ],
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "<pad>": 0,
                        "<eos>": 1,
                        "<unk>": 2,
                        "a": 3,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = read_hf_json_tokenizer_metadata(tokenizer_path)

    assert metadata["vocab_size"] == 10
    assert metadata["pad_id"] == 0
    assert metadata["unk_id"] == 2
    assert metadata["eos_id"] == 1
