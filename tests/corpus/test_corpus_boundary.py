from __future__ import annotations

import sys
import types
from pathlib import Path

import jax.numpy as jnp
import pytest
import yaml

from mintdim_lab.corpus import Batch, from_unit_read, stack_batches, to_jax_batch
from mintdim_lab.corpus.config_paths import (
    resolve_unit_build_config_path,
    resolve_unit_read_config_path,
)
from mintdim_lab.corpus.formatting import prompt_answer_prefix
from mintdim_lab.corpus.mintdim_unit_build import run_unit_build_config, unit_build_pipeline
from mintdim_lab.corpus.mintdim_unit_read import unit_read_pipeline, unit_read_records
from mintdim_lab.trainer.state import Batch as TrainingBatch

REPO = Path(__file__).resolve().parents[2]


def test_corpus_owns_batch_type_and_training_reexports_same_type():
    record = {
        "input_ids": [[1, 2], [3, 4]],
        "target_ids": [[2, 1], [4, 1]],
        "target_mask": [[1, 1], [1, 1]],
    }

    batch = to_jax_batch(from_unit_read(record))
    stacked = stack_batches([batch, batch])

    assert Batch is TrainingBatch
    assert isinstance(batch, Batch)
    assert stacked.input_ids.shape == (2, 2, 2)
    assert int(jnp.sum(stacked.target_mask)) == 8


def test_corpus_formatting_contract():
    assert prompt_answer_prefix("2+5+1=") == "#Prompt_user: 2+5+1=\n#Answer:\n"


def test_corpus_config_directory_requires_canonical_file_names(tmp_path):
    corpus_dir = tmp_path / "custom_corpus"
    corpus_dir.mkdir()
    explicit_file = corpus_dir / "custom.yaml"
    explicit_file.write_text("x: 1\n", encoding="utf-8")

    assert resolve_unit_build_config_path(explicit_file) == explicit_file
    assert resolve_unit_read_config_path(explicit_file) == explicit_file

    with pytest.raises(FileNotFoundError, match="unit_build.yaml"):
        resolve_unit_build_config_path(corpus_dir)

    with pytest.raises(FileNotFoundError, match="unit_read.yaml"):
        resolve_unit_read_config_path(corpus_dir)


def test_unit_build_adapter_exposes_public_mintdim_pipeline(monkeypatch):
    calls = []

    def pipeline(name: str):
        calls.append(name)
        return "unit-build-pipeline"

    monkeypatch.setitem(sys.modules, "mintdim", types.SimpleNamespace(pipeline=pipeline))

    assert unit_build_pipeline() == "unit-build-pipeline"
    assert calls == ["unit-build"]


def test_unit_build_config_runs_public_mintdim_pipeline_shape(monkeypatch):
    calls = []

    class FakeSource:
        def jsonl(self, *, files, fields, templates):
            calls.append(("source.jsonl", files, fields, templates))
            return fake_pipeline

    class FakeTokenizer:
        def hf_json(self, *, files):
            calls.append(("tokenizer.hf_json", files))
            return fake_pipeline

    class FakeOutput:
        def dir(self, path, *, samples_per_shard):
            calls.append(("output.dir", path, samples_per_shard))
            return fake_pipeline

    class FakePipeline:
        def __init__(self):
            self.source = FakeSource()
            self.tokenizer = FakeTokenizer()
            self.output = FakeOutput()

        def units(self, *, sizes, build_batch):
            calls.append(("units", sizes, build_batch))
            return self

        def run(self, **kwargs):
            calls.append(("run", kwargs))
            return {"outputs": []}

    fake_pipeline = FakePipeline()

    def pipeline(name: str):
        calls.append(("pipeline", name))
        return fake_pipeline

    monkeypatch.setitem(sys.modules, "mintdim", types.SimpleNamespace(pipeline=pipeline))

    result = run_unit_build_config(
        {
            "source": {
                "jsonl": {
                    "files": ["a.jsonl"],
                    "fields": [["prompt", "target"]],
                    "templates": ["{prompt}\\n{target}<eos>"],
                }
            },
            "tokenizer": {"hf_json": {"files": ["tokenizer.json"]}},
            "units": {"sizes": [[96]], "build_batch": [64]},
            "output": {
                "dir": {
                    "path": "out",
                    "samples_per_shard": [1500],
                }
            },
            "run": {"on_overflow": "abort"},
        }
    )

    assert result == {"outputs": []}
    assert calls == [
        ("pipeline", "unit-build"),
        ("source.jsonl", ["a.jsonl"], [["prompt", "target"]], ["{prompt}\\n{target}<eos>"]),
        ("tokenizer.hf_json", ["tokenizer.json"]),
        ("units", [[96]], [64]),
        ("output.dir", "out", [1500]),
        ("run", {"on_overflow": "abort"}),
    ]


def test_unit_read_adapter_exposes_public_mintdim_pipeline(monkeypatch):
    calls = []

    def pipeline(name: str):
        calls.append(name)
        return "unit-read-pipeline"

    monkeypatch.setitem(sys.modules, "mintdim", types.SimpleNamespace(pipeline=pipeline))

    assert unit_read_pipeline() == "unit-read-pipeline"
    assert calls == ["unit-read"]


def test_unit_read_adapter_uses_public_mintdim_pipeline_shape(monkeypatch):
    calls = []

    class FakeUnitReadPipeline:
        def __init__(self):
            self.read_calls = 0

        def queue(self, *, entries):
            calls.append(("queue", entries))
            return self

        def layout(self, *, sequence_template):
            calls.append(("layout", sequence_template))
            return self

        def target(self, *, fields, index, ignore_id=-100):
            calls.append(("target", fields, index, ignore_id))
            return self

        def read(self):
            self.read_calls += 1
            return {"read_calls": self.read_calls}

    fake_reader = FakeUnitReadPipeline()

    def pipeline(name: str):
        calls.append(("pipeline", name))
        return fake_reader

    monkeypatch.setitem(sys.modules, "mintdim", types.SimpleNamespace(pipeline=pipeline))

    records = unit_read_records(
        entries=[{"path": "./unit_96", "unit": 96, "batch": 2, "accum": 3}],
        sequence_template=["prompt", "token_template", "answer"],
        target_fields=("answer",),
        target_index=(2,),
        ignore_id=-9,
    )

    assert next(records) == {"read_calls": 1}
    assert next(records) == {"read_calls": 2}
    assert calls == [
        ("pipeline", "unit-read"),
        ("queue", [{"path": "./unit_96", "unit": 96, "batch": 2, "accum": 3}]),
        ("layout", ("prompt", "token_template", "answer")),
        ("target", ["answer"], [2], -9),
    ]


def test_corpus_keeps_no_manifest_or_schema_source_modules():
    corpus_root = REPO / "src" / "mintdim_lab" / "corpus"

    assert not (corpus_root / "manifest.py").exists()
    assert not (corpus_root / "schema.py").exists()


def test_addition_3term_corpus_directory_uses_smallest_fitting_unit():
    corpus_dir = REPO / "recipes" / "corpus" / "addition_3term_unit19"
    unit_build = yaml.safe_load(
        (corpus_dir / "unit_build.yaml").read_text(encoding="utf-8")
    )
    unit_read = yaml.safe_load(
        (corpus_dir / "unit_read.yaml").read_text(encoding="utf-8")
    )

    assert unit_build["source"]["jsonl"]["files"] == [
        "./data_store/raw/addition_3term/train.jsonl"
    ]
    assert unit_build["source"]["jsonl"]["fields"] == [["prompt", "answer"]]
    assert unit_build["tokenizer"]["hf_json"]["files"] == [
        "./data_store/tokenizers/addition_3term_byte_bpe_512/tokenizer.json"
    ]
    assert unit_build["units"]["sizes"] == [[19]]
    assert unit_build["output"]["dir"]["path"] == (
        "./data_store/packed/addition_3term_unit19"
    )
    assert unit_read["queue"][0]["path"] == (
        "./data_store/packed/addition_3term_unit19/unit_19"
    )
    assert unit_read["queue"][0]["unit"] == 19
    assert unit_read["queue"][0]["batch"] == 50
    assert unit_read["queue"][0]["accum"] == 10
    assert unit_read["target"]["fields"] == ["answer", "token_template"]
