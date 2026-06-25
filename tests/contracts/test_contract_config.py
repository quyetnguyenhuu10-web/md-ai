from __future__ import annotations

from arch_contract.config import load_contract_config
from arch_contract.files import iter_source_files


def test_contract_config_is_valid():
    config = load_contract_config()

    assert config.source_roots
    assert config.layers
    assert iter_source_files(config), "architecture contract did not discover any source files"
