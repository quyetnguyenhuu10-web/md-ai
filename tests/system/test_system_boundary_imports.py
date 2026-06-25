from __future__ import annotations

from mintdim_lab.system import jax_cpu
from mintdim_lab.system.paths import (
    DEFAULT_VOCAB_PATH,
    PACKAGE_ROOT,
    REPO_ROOT,
    SYSTEM_PACKAGE_ROOT,
)


def test_system_paths_use_production_runtime_locations():
    assert PACKAGE_ROOT == SYSTEM_PACKAGE_ROOT
    assert DEFAULT_VOCAB_PATH == (
        REPO_ROOT / "data_store" / "tokenizers" / "byte_bpe_512" / "tokenizer.json"
    )
    assert DEFAULT_VOCAB_PATH.is_file()


def test_jax_cpu_runtime_boundary_points_at_system_runtime():
    assert jax_cpu.platform() == "cpu"
    assert callable(jax_cpu.compile_callable)
