from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mintdim_lab.cli.commands.train import main as train_main
from mintdim_lab.cli.main import main as md_main

TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def _small_unit_read_config(tmp_path: Path) -> Path:
    path = tmp_path / "unit_read_small.yaml"
    path.write_text(
        """queue:
  - path: ./data_store/packed/linear_equation_unit96/unit_96
    unit: 96
    batch: 2
    accum: 1
layout:
  sequence_template:
    - prompt
    - token_template
    - target
    - token_template
target:
  fields:
    - target
    - token_template
  index:
    - 2
    - 3
  ignore_id: -100
""",
        encoding="utf-8",
    )
    return path


def test_train_cli_runs_one_real_step_without_compile(capsys, tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    unit_read_config = _small_unit_read_config(tmp_path)

    code = train_main(
        [
            "--repo",
            str(repo),
            "--model-config",
            TEST_MODEL_CONFIG,
            "--unit-read-config",
            str(unit_read_config),
            "--max-steps",
            "1",
            "--no-compile-update",
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["runtime"] == "cpu"
    assert payload["config_paths"]["model_config"].endswith(".yaml")
    assert (
        payload["config_paths"]["tokenizer_config"]
        == "recipes/tokenizers/byte_bpe_512.yaml"
    )
    assert (
        payload["config_paths"]["unit_read_config"]
        == str(unit_read_config)
    )
    assert (
        payload["config_paths"]["checkpoint_dir"]
        == "runs/train/linear_equation_tiny_cpu/checkpoints"
    )
    assert payload["config_paths"]["checkpoint_max_to_keep"] == 15
    assert payload["config_paths"]["checkpoint_dir_override"] is None
    assert payload["config_paths"]["model_config_override"].endswith(
        "recipes\\models\\test_tiny.yaml"
    ) or payload["config_paths"]["model_config_override"].endswith(
        "recipes/models/test_tiny.yaml"
    )
    assert payload["config_paths"]["tokenizer_config_override"] is None
    assert payload["config_paths"]["unit_read_config_override"] == str(unit_read_config)
    assert payload["max_steps"] == 1
    assert payload["seed"] == 0
    assert payload["final_step"] == 1
    assert payload["gradient_accumulation_steps"] == 1
    assert payload["unit_read"]["target_fields"] == ["target", "token_template"]
    assert payload["unit_read"]["target_index"] == [2, 3]
    assert payload["last_metrics"]["effective_batch_size"] == 2
    assert payload["last_metrics"]["token_count"] > 0.0


def test_train_cli_writes_optional_determinism_log(capsys, tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    unit_read_config = _small_unit_read_config(tmp_path)
    debug_path = tmp_path / "determinism.jsonl"

    code = train_main(
        [
            "--repo",
            str(repo),
            "--model-config",
            TEST_MODEL_CONFIG,
            "--unit-read-config",
            str(unit_read_config),
            "--runtime",
            "cpu",
            "--max-steps",
            "1",
            "--no-compile-update",
            "--determinism-log",
            str(debug_path),
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)
    assert payload["determinism_log"] == str(debug_path)

    records = [json.loads(line) for line in debug_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["run", "init", "step"]
    assert records[0]["seed"] == 0
    assert records[0]["runtime"]["runtime"] == "cpu"
    assert records[0]["unit_read_sources"][0]["files"][0]["sha256"]
    assert records[1]["step"] == 0
    assert len(records[1]["params_sha256"]) == 64
    assert records[2]["step"] == 1
    assert records[2]["effective_batch_size"] == 2
    assert len(records[2]["batch_sha256"]) == 64
    assert len(records[2]["accum_sha256"]) == 64
    assert len(records[2]["params_sha256"]) == 64
    assert len(records[2]["opt_state_sha256"]) == 64


def test_top_level_cli_dispatches_train(capsys, tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    unit_read_config = _small_unit_read_config(tmp_path)

    code = md_main(
        [
            "train",
            "--repo",
            str(repo),
            "--model-config",
            TEST_MODEL_CONFIG,
            "--unit-read-config",
            str(unit_read_config),
            "--runtime",
            "cpu",
            "--max-steps",
            "1",
            "--no-compile-update",
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["final_step"] == 1


def test_root_cli_dispatches_train_without_install_or_env(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    unit_read_config = _small_unit_read_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "src/mintdim_lab/cli/main.py",
            "train",
            "--repo",
            str(repo),
            "--model-config",
            TEST_MODEL_CONFIG,
            "--unit-read-config",
            str(unit_read_config),
            "--runtime",
            "cpu",
            "--max-steps",
            "1",
            "--no-compile-update",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""

    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["final_step"] == 1


def test_root_cli_train_command_runs_directly(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    unit_read_config = _small_unit_read_config(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "src/mintdim_lab/cli/main.py",
            "train",
            "--repo",
            str(repo),
            "--model-config",
            TEST_MODEL_CONFIG,
            "--unit-read-config",
            str(unit_read_config),
            "--runtime",
            "cpu",
            "--max-steps",
            "1",
            "--no-compile-update",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""

    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["final_step"] == 1


def test_train_cli_rejects_bad_max_steps(capsys):
    repo = Path(__file__).resolve().parents[2]

    code = train_main(
        [
            "--repo",
            str(repo),
            "--runtime",
            "cpu",
            "--max-steps",
            "0",
            "--no-compile-update",
        ]
    )

    captured = capsys.readouterr()

    assert code == 1
    assert "--max-steps must be positive" in captured.err


def test_train_cli_seed_override_wins_over_training_yaml(capsys, tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    unit_read_config = _small_unit_read_config(tmp_path)

    code = train_main(
        [
            "--repo",
            str(repo),
            "--model-config",
            TEST_MODEL_CONFIG,
            "--unit-read-config",
            str(unit_read_config),
            "--runtime",
            "cpu",
            "--max-steps",
            "1",
            "--seed",
            "7",
            "--no-compile-update",
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""
    assert json.loads(captured.out)["seed"] == 7


def test_train_cli_allows_config_path_overrides(capsys, tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    unit_read_config = _small_unit_read_config(tmp_path)

    code = train_main(
        [
            "--repo",
            str(repo),
            "--training-config",
            "recipes/train/cpu/linear_equation_tiny_cpu.yaml",
            "--model-config",
            TEST_MODEL_CONFIG,
            "--tokenizer-config",
            "recipes/tokenizers/byte_bpe_512.yaml",
            "--unit-read-config",
            str(unit_read_config),
            "--runtime",
            "cpu",
            "--max-steps",
            "1",
            "--no-compile-update",
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["config_paths"]["model_config_override"].endswith(
        "recipes\\models\\test_tiny.yaml"
    ) or payload["config_paths"]["model_config_override"].endswith("recipes/models/test_tiny.yaml")
    assert payload["config_paths"]["tokenizer_config_override"].endswith(
        "recipes\\tokenizers\\byte_bpe_512.yaml"
    ) or payload["config_paths"]["tokenizer_config_override"].endswith(
        "recipes/tokenizers/byte_bpe_512.yaml"
    )
    assert payload["config_paths"]["unit_read_config_override"].endswith(str(unit_read_config))
    assert payload["final_step"] == 1


def test_train_cli_saves_orbax_checkpoint_on_save_every(tmp_path, capsys):
    repo = Path(__file__).resolve().parents[2]
    unit_read_config = _small_unit_read_config(tmp_path)

    base = repo / "recipes" / "train" / "cpu" / "linear_equation_tiny_cpu.yaml"
    training_config = tmp_path / "basic_save_every_1.yaml"
    text = base.read_text(encoding="utf-8")
    text = text.replace("save_every: 100", "save_every: 1")
    text = text.replace(
        "checkpoint_dir: runs/train/linear_equation_tiny_cpu/checkpoints",
        f"checkpoint_dir: {tmp_path.as_posix()}/checkpoints",
    )
    training_config.write_text(text, encoding="utf-8")

    code = train_main(
        [
            "--repo",
            str(repo),
            "--training-config",
            str(training_config),
            "--model-config",
            TEST_MODEL_CONFIG,
            "--unit-read-config",
            str(unit_read_config),
            "--runtime",
            "cpu",
            "--max-steps",
            "1",
            "--no-compile-update",
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)
    assert payload["checkpoint"]["saved_steps"] == [1]
    assert Path(payload["checkpoint"]["dir"]).is_dir()
    assert (Path(payload["checkpoint"]["dir"]) / "step_00000001").is_dir()


def test_train_cli_checkpoint_metadata_includes_determinism_debug_when_enabled(
    tmp_path,
    capsys,
):
    repo = Path(__file__).resolve().parents[2]
    unit_read_config = _small_unit_read_config(tmp_path)
    debug_path = tmp_path / "determinism.jsonl"

    base = repo / "recipes" / "train" / "cpu" / "linear_equation_tiny_cpu.yaml"
    training_config = tmp_path / "basic_save_every_1.yaml"
    text = base.read_text(encoding="utf-8")
    text = text.replace("save_every: 100", "save_every: 1")
    text = text.replace(
        "checkpoint_dir: runs/train/linear_equation_tiny_cpu/checkpoints",
        f"checkpoint_dir: {tmp_path.as_posix()}/checkpoints",
    )
    training_config.write_text(text, encoding="utf-8")

    code = train_main(
        [
            "--repo",
            str(repo),
            "--training-config",
            str(training_config),
            "--model-config",
            TEST_MODEL_CONFIG,
            "--unit-read-config",
            str(unit_read_config),
            "--runtime",
            "cpu",
            "--max-steps",
            "1",
            "--no-compile-update",
            "--determinism-log",
            str(debug_path),
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)
    checkpoint = Path(payload["checkpoint"]["dir"]) / "step_00000001"
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    debug = metadata["debug"]["determinism"]
    assert debug["enabled"] is True
    assert debug["step"] == 1
    assert debug["log_path"].endswith("determinism.jsonl")
    assert len(debug["params_sha256"]) == 64
    assert len(debug["opt_state_sha256"]) == 64
