from __future__ import annotations

from arch_contract.config import load_contract_config
from arch_contract.imports import collect_python_imports
from arch_contract.rules import (
    assert_no_unbaselined_violations,
    runtime_wiring_violations,
)


def test_runtime_wiring_only_in_composition_root():
    config = load_contract_config()
    imports = collect_python_imports(config)
    violations = runtime_wiring_violations(config, imports)

    assert_no_unbaselined_violations(violations, config)
