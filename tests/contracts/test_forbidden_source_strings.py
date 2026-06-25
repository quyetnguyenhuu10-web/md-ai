from __future__ import annotations

from arch_contract.config import load_contract_config
from arch_contract.rules import (
    assert_no_unbaselined_violations,
    forbidden_source_string_violations,
)


def test_forbidden_source_strings():
    config = load_contract_config()
    violations = forbidden_source_string_violations(config)

    assert_no_unbaselined_violations(violations, config)
