from __future__ import annotations

from arch_contract.config import load_contract_config
from arch_contract.graph import cycle_violations
from arch_contract.imports import collect_python_imports
from arch_contract.rules import assert_no_unbaselined_violations


def test_no_project_import_cycles():
    config = load_contract_config()
    imports = collect_python_imports(config)
    violations = cycle_violations(config, imports)

    assert_no_unbaselined_violations(violations, config)
