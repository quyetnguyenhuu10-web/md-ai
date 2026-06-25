from __future__ import annotations

from pathlib import Path

from mintdim_lab.evaluator.config import load_sectioned_config
from mintdim_lab.evaluator.config_schema import SECTION_FIELDS
from mintdim_lab.run_setup.bundle import load_training_bundle
from mintdim_lab.tokenizer.config import load_tokenizer_config_yaml
from mintdim_lab.trainer.config import (
    load_training_config_yaml,
    load_unit_read_config_yaml,
)

REPO = Path(__file__).resolve().parents[2]
TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def test_phase2_recipes_resolve_training_corpus_and_tokenizer_configs():
    training = load_training_config_yaml(
        REPO / "recipes" / "train" / "gpu" / "linear_equation_tiny_gpu.yaml"
    )
    tokenizer = load_tokenizer_config_yaml(REPO / training.tokenizer_config)
    unit_read = load_unit_read_config_yaml(REPO / training.unit_read_config)

    assert training.model_config == "recipes/models/tiny.yaml"
    assert training.tokenizer_config == "recipes/tokenizers/byte_bpe_512.yaml"
    assert (
        training.unit_read_config
        == "recipes/corpus/linear_equation_unit96/unit_read.yaml"
    )
    assert training.checkpoint_dir == "runs/train/linear_equation_tiny_gpu/checkpoints"
    assert training.runtime.name == "gpu"
    assert training.runtime.device_index == 0
    assert training.runtime.global_device is False
    assert training.runtime.compile_update is True
    assert tokenizer.path == "./data_store/tokenizers/byte_bpe_512/tokenizer.json"
    assert tokenizer.vocab_size == 512
    assert tuple(entry.path for entry in unit_read.queue) == (
        "./data_store/packed/linear_equation_unit96/unit_96",
    )
    assert tuple(entry.unit for entry in unit_read.queue) == (96,)
    assert unit_read.target_index == (2, 3)


def test_phase2_training_bundle_can_use_recipe_paths_with_test_model():
    bundle = load_training_bundle(
        repo=REPO,
        training_config="recipes/train/gpu/linear_equation_tiny_gpu.yaml",
        model_config=TEST_MODEL_CONFIG,
    )

    assert bundle.training.tokenizer_config == "recipes/tokenizers/byte_bpe_512.yaml"
    assert (
        bundle.training.unit_read_config
        == "recipes/corpus/linear_equation_unit96/unit_read.yaml"
    )
    assert bundle.model_config.num_layers == 2
    assert bundle.model_config.max_seq_len == 128
    assert bundle.tokenizer.path == "./data_store/tokenizers/byte_bpe_512/tokenizer.json"
    assert bundle.unit_read.queue[0].path == "./data_store/packed/linear_equation_unit96/unit_96"


def test_phase2_evaluation_recipe_points_at_data_store():
    config = load_sectioned_config(
        REPO / "recipes" / "evaluation" / "math_exact.yaml",
        section_fields=SECTION_FIELDS,
    )

    assert config["eval_path"] == "data_store/raw/linear_equation/eval.jsonl"
    assert config["vocab_path"] == "data_store/tokenizers/byte_bpe_512/tokenizer.json"
    assert (REPO / config["eval_path"]).is_file()
    assert (REPO / config["vocab_path"]).is_file()


def test_phase2_data_store_contains_raw_packed_and_tokenizer_artifacts():
    paths = [
        REPO / "data_store" / "raw" / "linear_equation" / "pretrain_sft.jsonl",
        REPO / "data_store" / "raw" / "linear_equation" / "eval.jsonl",
        REPO / "data_store" / "packed" / "linear_equation_unit96" / "manifest.json",
        REPO / "data_store" / "packed" / "linear_equation_unit96" / "stats.json",
        REPO / "data_store" / "packed" / "linear_equation_unit96" / "unit_96" / "shard_000000.bin",
        REPO / "data_store" / "tokenizers" / "byte_bpe_512" / "tokenizer.json",
        REPO / "data_store" / "tokenizers" / "byte_bpe_512" / "tokenizer_unk.jsonl",
    ]

    for path in paths:
        assert path.is_file(), path
