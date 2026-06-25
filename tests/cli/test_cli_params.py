from __future__ import annotations

import json
from pathlib import Path

from mintdim_lab.cli.commands.params import main as params_main
from mintdim_lab.cli.main import main as md_main

TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def test_params_cli_counts_test_tiny_with_repo_tokenizer(capsys):
    repo = Path(__file__).resolve().parents[2]

    code = params_main(
        [
            "--repo",
            str(repo),
            "--model-config",
            TEST_MODEL_CONFIG,
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["model_config"] == TEST_MODEL_CONFIG
    assert payload["tokenizer_config"] == "recipes/tokenizers/byte_bpe_512.yaml"
    assert payload["tokenizer"]["vocab_size"] == 512
    assert payload["model"]["num_embed"] == 512
    assert payload["model"]["params"] == 36_128
    assert payload["model"]["params_m"] == 0.036128
    assert payload["model"]["ffw_activation"] == "swiglu"
    assert payload["model"]["ffn_ratio"] == 2.0


def test_top_level_cli_dispatches_params(capsys):
    repo = Path(__file__).resolve().parents[2]

    code = md_main(["params", "--repo", str(repo), "--model-config", TEST_MODEL_CONFIG])

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["model_config"] == TEST_MODEL_CONFIG
    assert payload["model"]["params"] > 0


def test_params_cli_resolves_tokenizer_from_repo_outside_working_directory(
    tmp_path,
    monkeypatch,
    capsys,
):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)

    code = params_main(
        [
            "--repo",
            str(repo),
            "--model-config",
            TEST_MODEL_CONFIG,
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""
    assert Path.cwd() == tmp_path
