from __future__ import annotations

import importlib

from arch_contract.config import load_contract_config
from arch_contract.files import relpath
from arch_contract.imports import build_python_module_index
from arch_contract.rules import (
    Violation,
    assert_no_unbaselined_violations,
    parse_static_all,
)


def test_public_api_contract():
    config = load_contract_config()
    violations: list[Violation] = []

    for module_name, module_info in build_python_module_index(config).items():
        exported = parse_static_all(module_info.path)
        if exported is None:
            continue
        if any(name.startswith("_") for name in exported):
            violations.append(
                Violation(
                    rule="public_api_private_export",
                    path=relpath(module_info.path, config.repo_root),
                    line=1,
                    detail=f"module={module_name} __all__ exports private names={exported}",
                )
            )

    for contract in config.data.get("public_api_contracts", []):
        module_name = contract["module"]
        mode = contract.get("mode", "minimum")
        expected = set(contract.get("expected", []))
        module = importlib.import_module(module_name)
        actual = set(getattr(module, "__all__", [name for name in dir(module) if not name.startswith("_")]))
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            violations.append(
                Violation(
                    rule="public_api_missing",
                    path=f"<module:{module_name}>",
                    line=0,
                    detail=f"missing={missing}",
                )
            )
        if mode == "exact" and extra:
            violations.append(
                Violation(
                    rule="public_api_extra",
                    path=f"<module:{module_name}>",
                    line=0,
                    detail=f"extra={extra}",
                )
            )

    assert_no_unbaselined_violations(violations, config)
