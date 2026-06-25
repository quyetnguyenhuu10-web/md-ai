from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PRODUCTION_MODULES = [
    "mintdim_lab.inference.checkpoint_selection",
    "mintdim_lab.inference.text_generator",
    "mintdim_lab.inference.kv_cache",
    "mintdim_lab.inference.model_loader",
    "mintdim_lab.system.checkpoint_io",
    "mintdim_lab.system.paths",
    "mintdim_lab.serving.prompting",
    "mintdim_lab.serving.http_api",
    "mintdim_lab.serving.worker_cli",
    "mintdim_lab.serving.worker_protocol",
    "mintdim_lab.serving.chat_session",
    "mintdim_lab.cli",
    "mintdim_lab.cli.main",
    "mintdim_lab.cli.commands.evaluate",
    "mintdim_lab.cli.commands.chat",
    "mintdim_lab.cli.commands.params",
    "mintdim_lab.cli.commands.train",
    "mintdim_lab.evaluator.config",
    "mintdim_lab.evaluator.config_schema",
    "mintdim_lab.evaluator.run_evaluation",
    "mintdim_lab.evaluator.data",
    "mintdim_lab.evaluator.generation",
    "mintdim_lab.evaluator.scoring",
    "mintdim_lab.evaluator.report",
    "mintdim_lab.evaluator.template",
    "mintdim_lab.trainer.config",
    "mintdim_lab.trainer.config.shared",
    "mintdim_lab.trainer.config.training",
    "mintdim_lab.trainer.config.unit_read",
    "mintdim_lab.trainer.run_training",
]


def test_legacy_md_package_is_not_importable():
    assert importlib.util.find_spec("md") is None


def test_legacy_md_directory_is_removed():
    assert not (REPO / "md").exists()


def test_production_boundaries_replace_legacy_modules():
    for module_name in PRODUCTION_MODULES:
        importlib.import_module(module_name)


def test_current_orbax_checkpoint_root_can_be_discovered_if_present():
    from mintdim_lab.system.checkpoint_io import find_checkpoints

    checkpoint_root = REPO / "runs" / "checkpoints"

    if checkpoint_root.exists():
        found = find_checkpoints(checkpoint_root)
        assert all(path.exists() for path in found)


def test_inference_runtime_uses_text_metadata_without_registry_side_effects():
    import mintdim_lab.inference.model_loader as runtime

    fields = runtime.REBUILDABLE_MODEL_METADATA_FIELDS
    assert "num_embed" in fields
    assert "max_seq_len" in fields
    assert "num_layers" in fields


def test_no_python_sources_import_legacy_md_package():
    roots = [REPO / "src", REPO / "tests", REPO / "behavior_locks"]
    offenders: list[str] = []
    legacy_package = "m" + "d"
    forbidden_prefixes = (
        f"from {legacy_package}.",
        f"import {legacy_package}.",
    )

    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith(forbidden_prefixes):
                    relative = path.relative_to(REPO).as_posix()
                    offenders.append(f"{relative}:{line_no}: {line}")

    assert offenders == []
