from __future__ import annotations

from pathlib import Path

from mintdim_lab.tokenizer import load_tokenizer
from mintdim_lab.tokenizer.config import (
    load_tokenizer_config_yaml,
    read_hf_json_tokenizer_metadata,
)

REPO = Path(__file__).resolve().parents[2]


def test_tokenizer_config_derives_hf_json_metadata_contract():
    cfg = load_tokenizer_config_yaml(REPO / "recipes" / "tokenizers" / "byte_bpe_512.yaml")
    tokenizer_path = (
        REPO / "data_store" / "tokenizers" / "byte_bpe_512" / "tokenizer.json"
    )
    metadata = read_hf_json_tokenizer_metadata(tokenizer_path)

    assert cfg.type == "hf_json"
    assert cfg.path == "./data_store/tokenizers/byte_bpe_512/tokenizer.json"
    assert cfg.vocab_size == 512
    assert cfg.pad_id == 0
    assert cfg.eos_id == 1
    assert cfg.unk_id == 2
    assert metadata == {"vocab_size": 512, "pad_id": 0, "unk_id": 2, "eos_id": 1}


def test_runtime_tokenizer_roundtrip_and_eos_append_contract():
    tokenizer_path = (
        REPO / "data_store" / "tokenizers" / "byte_bpe_512" / "tokenizer.json"
    )
    tokenizer = load_tokenizer(tokenizer_path)

    plain_ids = tokenizer.encode("Giải phương trình: 2x-2=3?", add_eos=False)
    eos_ids = tokenizer.encode("Giải phương trình: 2x-2=3?", add_eos=True)

    assert plain_ids
    assert eos_ids[:-1] == plain_ids
    assert eos_ids[-1] == tokenizer.eos_id
    assert "2x" in tokenizer.decode(plain_ids)
