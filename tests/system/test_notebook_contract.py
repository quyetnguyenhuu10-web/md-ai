from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GPU_NOTEBOOK = REPO / "studies" / "notebooks" / "pretraining_gpu_20_seeds.ipynb"
TPU_NOTEBOOK = REPO / "studies" / "notebooks" / "pretraining_tpu_20_seeds.ipynb"


def test_pretraining_gpu_notebook_uses_linear_equation_paths():
    required = (
        "mintdim_lab.cli.commands.train",
        "mintdim_lab.evaluator.run_evaluation",
        "recipes/train/gpu/linear_equation_tiny_gpu.yaml",
        "recipes/evaluation/math_exact.yaml",
        "data_store/packed/linear_equation_unit96/unit_96",
        "data_store/raw/linear_equation/eval.jsonl",
        "data_store/tokenizers/byte_bpe_512/tokenizer.json",
        "math_exact.json",
    )
    retired = (
        "recipes/train/gpu/addition_3term_tiny_gpu.yaml",
        "recipes/evaluation/addition_3term_exact.yaml",
        "data_store/packed/addition_3term_unit19/unit_19",
        "data_store/raw/addition_3term/eval.jsonl",
        "data_store/tokenizers/addition_3term_byte_bpe_512/tokenizer.json",
        "addition_3term_exact.json",
    )

    body = GPU_NOTEBOOK.read_text(encoding="utf-8")
    for needle in required:
        assert needle in body, f"{GPU_NOTEBOOK} missing current notebook contract: {needle}"
    for needle in retired:
        assert needle not in body, f"{GPU_NOTEBOOK} still references retired path: {needle}"


def test_pretraining_tpu_notebook_still_uses_linear_equation_paths():
    required = (
        "mintdim_lab.cli.commands.train",
        "mintdim_lab.evaluator.run_evaluation",
        "recipes/evaluation/math_exact.yaml",
        "data_store/packed/linear_equation_unit96/unit_96",
        "data_store/tokenizers/byte_bpe_512/tokenizer.json",
    )

    body = TPU_NOTEBOOK.read_text(encoding="utf-8")
    for needle in required:
        assert needle in body, f"{TPU_NOTEBOOK} missing current notebook contract: {needle}"


def test_pretraining_notebooks_do_not_use_retired_paths():
    retired = (
        "md.cli",
        "md.benchmark",
        "from md",
        "configs/training",
        "configs/benchmark",
        "md/data",
        "md/vocab",
        "recipes/runtimes",
        "recipes/experiments",
        "--runtime",
    )

    for path in (GPU_NOTEBOOK, TPU_NOTEBOOK):
        body = path.read_text(encoding="utf-8")
        for needle in retired:
            assert needle not in body, f"{path} still references retired path: {needle}"
