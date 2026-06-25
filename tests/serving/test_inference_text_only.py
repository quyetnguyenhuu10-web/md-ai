from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_inference_directory_has_no_multimodal_input_module():
    assert not (REPO / "md" / "inference" / "multimodal_input.py").exists()


def test_generate_is_text_only():
    body = text("src/mintdim_lab/inference/text_generator.py")

    assert "image_path" not in body
    assert "multimodal" not in body.lower()
    assert "vision" not in body.lower()
    assert "_prepare_image_inputs" not in body
    assert "_prefill_apply_for_multimodal_model" not in body
    assert "encode_multimodal_prompt" not in body
    assert "_prefill_apply_for_model" in body
    assert "_decode_apply_for_model" in body


def test_runtime_is_text_only():
    body = text("src/mintdim_lab/inference/model_loader.py")

    assert "MultimodalConfig" not in body
    assert "VisionConfig" not in body
    assert "MultimodalTransformer" not in body
    assert "_vision_config_from_metadata" not in body
    assert "_multimodal_config_from_metadata" not in body
    assert "LEGACY_PRESET_ALIASES" not in body
    assert "checkpoint_preset" not in body
    assert "def build_model_and_generate_fn(*, config: Any)" in body
    assert "return Transformer(config), generate" in body


def test_checkpoint_selection_accepts_current_orbax_step_dirs():
    body = text("src/mintdim_lab/inference/checkpoint_selection.py")

    assert 'candidate.name.startswith("step_")' in body
    assert '(candidate / "params").is_dir()' not in body
