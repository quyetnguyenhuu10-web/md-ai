from __future__ import annotations

from arch_contract.config import load_contract_config
from arch_contract.imports import collect_python_imports
from arch_contract.rules import (
    assert_no_unbaselined_violations,
    layer_import_boundary_violations,
)


def test_layer_import_boundaries_are_respected():
    config = load_contract_config()
    imports = collect_python_imports(config)
    violations = layer_import_boundary_violations(config, imports)

    assert_no_unbaselined_violations(violations, config)
