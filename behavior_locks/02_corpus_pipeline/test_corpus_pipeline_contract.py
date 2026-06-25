from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from mintdim_lab.corpus.batch import Batch, from_unit_read, stack_batches, to_jax_batch
from mintdim_lab.trainer.config import build_unit_read_config, load_unit_read_config_yaml

REPO = Path(__file__).resolve().parents[2]


def test_unit_read_yaml_contract_for_current_training_corpus():
    cfg = load_unit_read_config_yaml(
        REPO / "recipes" / "corpus" / "linear_equation_unit96" / "unit_read.yaml"
    )

    assert cfg.entries == [
        {
            "path": "./data_store/packed/linear_equation_unit96/unit_96",
            "unit": 96,
            "batch": 50,
            "accum": 30,
        }
    ]
    assert cfg.sequence_template == ("prompt", "token_template", "target", "token_template")
    assert cfg.target_fields == ("target", "token_template")
    assert cfg.target_index == (2, 3)
    assert cfg.ignore_id == -100
    assert cfg.to_unit_read_kwargs()["target_index"] == (2, 3)


def test_unit_read_target_index_is_zero_based_and_segment_matched():
    with pytest.raises(ValueError, match="0-base"):
        build_unit_read_config(
            {
                "queue": [{"path": "./data/unit_16", "unit": 16, "batch": 2, "accum": 1}],
                "layout": {"sequence_template": ["prompt", "token_template", "target"]},
                "target": {"fields": ["target"], "index": [3], "ignore_id": -100},
            }
        )

    with pytest.raises(ValueError, match="field/index pairs"):
        build_unit_read_config(
            {
                "queue": [{"path": "./data/unit_16", "unit": 16, "batch": 2, "accum": 1}],
                "layout": {"sequence_template": ["prompt", "token_template", "target"]},
                "target": {"fields": ["target"], "index": [1], "ignore_id": -100},
            }
        )


def test_batch_conversion_and_accumulation_window_shape_contract():
    record = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "target_ids": [[2, 3, 1], [5, 6, 1]],
        "target_mask": [[0, 1, 1], [0, 1, 1]],
    }

    batch = to_jax_batch(from_unit_read(record))
    stacked = stack_batches([batch, batch])

    assert isinstance(batch, Batch)
    assert batch.input_ids.shape == (2, 3)
    assert batch.target_ids.shape == (2, 3)
    assert batch.target_mask.shape == (2, 3)
    assert stacked.input_ids.shape == (2, 2, 3)
    assert stacked.target_ids.shape == (2, 2, 3)
    assert stacked.target_mask.shape == (2, 2, 3)
    assert int(jnp.sum(stacked.target_mask)) == 8


def test_corpus_source_keeps_manifest_and_schema_out_of_runtime_boundary():
    corpus_root = REPO / "src" / "mintdim_lab" / "corpus"

    assert not (corpus_root / "manifest.py").exists()
    assert not (corpus_root / "schema.py").exists()
