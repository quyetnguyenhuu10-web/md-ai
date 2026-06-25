from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_fused_attention_forces_cudnn_backend() -> None:
    source = text("src/mintdim_lab/model/attention.py")

    assert 'implementation="cudnn"' in source
    assert "jax.nn.dot_product_attention" in source
    assert "NVIDIA GPU/cuDNN flash attention only" in source
    assert "not a TPU backend" in source


def test_custom_flash_attention_doc_is_portable_runtime_contract() -> None:
    source = text("src/mintdim_lab/model/attention.py")

    assert "Portable custom FlashAttention via blockwise online softmax" in source
    assert "ordinary JAX/XLA primitives" in source
    assert "any runtime that can compile those primitives" in source
    assert "CPU, GPU, and TPU runtimes" in source
    assert "never materializes full float32 ``Q x K`` scores" in source


def test_config_doc_records_attention_backend_contracts() -> None:
    source = text("src/mintdim_lab/model/config.py")

    assert "use_flash_attention enables MintDim's portable custom blockwise" in source
    assert "use_fused_attention enables JAX dot_product_attention" in source
    assert 'implementation="cudnn"' in source
    assert "NVIDIA GPU/cuDNN flash attention only" in source
    assert "not a TPU backend" in source
