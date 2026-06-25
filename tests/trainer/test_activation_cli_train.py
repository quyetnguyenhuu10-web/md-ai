from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mintdim_lab.cli.commands.train import main as train_main

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


@pytest.mark.parametrize("activation", ["gelu", "geglu", "silu", "swiglu"])
def test_cli_model_config_override_trains_one_real_step_for_each_activation(
    activation: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = Path(__file__).resolve().parents[2]
    model_config = tmp_path / f"tiny-{activation}.yaml"
    metrics_path = tmp_path / f"metrics-{activation}.jsonl"
    unit_read_config = _small_unit_read_config(tmp_path)

    model_text = (repo / TEST_MODEL_CONFIG).read_text(encoding="utf-8")
    model_config.write_text(
        model_text.replace("activation: swiglu", f"activation: {activation}"),
        encoding="utf-8",
    )

    code = train_main(
        [
            "--repo",
            str(repo),
            "--model-config",
            str(model_config),
            "--unit-read-config",
            str(unit_read_config),
            "--runtime",
            "cpu",
            "--max-steps",
            "1",
            "--no-compile-update",
            "--jsonl",
            str(metrics_path),
        ]
    )

    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["final_step"] == 1
    assert payload["config_paths"]["model_config_override"] == str(model_config)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["step"] == 1
    assert metrics["effective_batch_size"] == 2
    assert metrics["token_count"] > 0.0
    assert math.isfinite(metrics["loss_mean"])
