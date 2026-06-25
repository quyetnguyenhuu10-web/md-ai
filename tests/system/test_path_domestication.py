from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_chat_checkpoint_defaults_are_repo_local():
    body = text("recipes/chat/checkpoints.yaml")

    assert "data_store/tokenizers/byte_bpe_512/tokenizer.json" in body

    assert "mintdim_lab" not in body
    assert "vocab.json" not in body


def test_chat_ui_spawns_repo_worker_module():
    body = text("apps/terminal/lib/worker.ts")

    assert '"src/mintdim_lab/cli/main.py", "chat", "--worker"' in body
    assert '"md.chat.worker"' not in body
    assert "mintdim.cli.chat.worker" not in body
    assert "mintdim_lab.cli.chat.worker" not in body


def test_foundation_paths_use_current_repo_defaults():
    body = text("src/mintdim_lab/system/paths.py")

    assert 'DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "runs" / "checkpoints"' in body
    assert 'DEFAULT_VOCAB_PATH = TOKENIZER_DIR / "tokenizer.json"' in body
    assert 'CHAT_CHECKPOINT_LIST_PATH = REPO_ROOT / "recipes" / "chat" / "checkpoints.yaml"' in body

    assert "LEGACY_CHECKPOINT_DIR" not in body
    assert "LM_DATA_DIR" not in body
    assert "HANDWRITING_OCR_DIR" not in body
    assert "FINEWEB_EDU_DIR" not in body
    assert "CHECKPOINT_METADATA_FIELDS" not in body
    assert 'TOKENIZER_DIR / "vocab.json"' not in body


def test_public_docs_and_scripts_do_not_pin_local_runtime_paths():
    forbidden = [
        "py " + "-3.14",
        "py " + "-3",
        "D:" + "\\",
        "C:" + "\\Users\\HP",
        "python" + "core",
        "." + "python312",
    ]

    for rel in (
        "README.md",
        "apps/terminal/package.json",
        "scripts/copy-clean-repo.ps1",
    ):
        body = text(rel)
        for needle in forbidden:
            assert needle not in body, f"{rel} contains local runtime hint: {needle}"


def test_public_docs_use_production_entrypoints_and_layout():
    readme = text("README.md")
    package_json = text("apps/terminal/package.json")

    assert "python src/mintdim_lab/cli/main.py train" in readme
    assert "recipes/chat/checkpoints.yaml" in readme
    assert "data_store/tokenizers/byte_bpe_512/tokenizer.json" in readme
    assert "apps/terminal" in readme
    assert not (REPO / "package.json").exists()
    legacy_cli_command = "python -m " + "md" + ".cli.main"
    assert legacy_cli_command not in readme
    assert "md/chat/ui" not in readme
    assert "md/chat/checkpoints.yaml" not in readme
    assert '"chat": "bun chat.tsx"' in package_json
    assert "md/chat/ui" not in package_json


def test_checkpoint_resolution_has_no_legacy_checkpoint_rewrite_shim():
    body = text("src/mintdim_lab/system/checkpoint_io.py")

    assert "_rewrite_legacy_checkpoint_path" not in body
    assert "_starts_with_legacy_relative_checkpoint" not in body
    assert "LEGACY_CHECKPOINT_DIR" not in body


def test_inference_server_owns_checkpoint_defaults_without_worker_import():
    body = text("src/mintdim_lab/serving/http_api.py")
    paths = text("src/mintdim_lab/system/paths.py")

    assert "md.chat.worker.checkpoints" not in body
    assert "def _default_checkpoint()" in body
    assert "CHAT_CHECKPOINT_LIST_PATH" in body
    assert (
        'CHAT_CHECKPOINT_LIST_PATH = REPO_ROOT / "recipes" / "chat" / "checkpoints.yaml"' in paths
    )
    assert "md/cli/chat/checkpoints.yaml" not in body
    assert "md/chat/checkpoints.yaml" not in body
    assert "vocab.json" not in body

    assert 'branch: str = "multimodal"' not in body
    assert '"multimodal"' not in body
    retired_model_hint = "Did you import " + "md" + ".nn?"
    assert retired_model_hint not in body


def test_legacy_md_directory_is_removed():
    assert not (REPO / "md").exists()
