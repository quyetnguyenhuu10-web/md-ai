from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

KNOWN_RULES = {
    "forbid_import_cycles",
    "forbid_private_cross_imports",
    "forbid_relative_parent_imports",
    "forbid_top_level_io",
    "forbid_runtime_wiring_outside_composition_root",
    "require_layer_markers",
    "strict_public_api",
    "allow_private_imports_for_tests",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "source_roots",
    "ignored_dirs",
    "layers",
    "allowed_imports",
    "forbidden_imports",
    "forbidden_source_strings",
    "public_api_contracts",
    "composition_roots",
    "runtime_wiring_imports",
    "rules",
}


@dataclass(frozen=True)
class ContractConfig:
    repo_root: Path
    data: dict[str, Any]
    baseline: set[str]

    @property
    def source_roots(self) -> list[str]:
        return list(self.data.get("source_roots", []))

    @property
    def ignored_dirs(self) -> list[str]:
        return list(self.data.get("ignored_dirs", []))

    @property
    def layers(self) -> dict[str, dict[str, Any]]:
        return dict(self.data.get("layers", {}))

    @property
    def rules(self) -> dict[str, bool]:
        return dict(self.data.get("rules", {}))


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "architecture_contract.yaml"
        ).is_file():
            return candidate
    raise RuntimeError("Could not find repo root with pyproject.toml and architecture_contract.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AssertionError(f"{path}: expected YAML mapping at top level")
    return value


def load_contract_config() -> ContractConfig:
    repo_root = find_repo_root()
    data = load_yaml(repo_root / "architecture_contract.yaml")
    validate_config_data(data)
    baseline = load_baseline(repo_root, data)
    return ContractConfig(repo_root=repo_root, data=data, baseline=baseline)


def load_baseline(repo_root: Path, data: dict[str, Any]) -> set[str]:
    baseline_config = data.get("baseline", {})
    if not isinstance(baseline_config, dict):
        return set()
    baseline_file = baseline_config.get("file")
    if not baseline_file:
        return set()
    baseline_data = load_yaml(repo_root / str(baseline_file))
    violations = baseline_data.get("violations", [])
    if violations is None:
        return set()
    if not isinstance(violations, list):
        raise AssertionError("architecture_contract_baseline.yaml: violations must be a list")
    return {str(item) for item in violations}


def validate_config_data(data: dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS - set(data)
    if missing:
        raise AssertionError(f"architecture_contract.yaml missing keys: {sorted(missing)}")

    for key in REQUIRED_TOP_LEVEL_KEYS:
        value = data.get(key)
        if value is None:
            raise AssertionError(f"architecture_contract.yaml: {key} must not be null")

    source_roots = data.get("source_roots")
    if not isinstance(source_roots, list) or not source_roots:
        raise AssertionError("architecture_contract.yaml: source_roots must be a non-empty list")
    for pattern in source_roots:
        assert_non_empty_string(pattern, "source_roots")

    ignored_dirs = data.get("ignored_dirs")
    if not isinstance(ignored_dirs, list):
        raise AssertionError("architecture_contract.yaml: ignored_dirs must be a list")
    for pattern in ignored_dirs:
        assert_non_empty_string(pattern, "ignored_dirs")

    layers = data.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise AssertionError("architecture_contract.yaml: layers must be a non-empty mapping")
    if len(layers) != len(set(layers)):
        raise AssertionError("architecture_contract.yaml: duplicate layer names are not allowed")
    for layer_name, layer in layers.items():
        assert_non_empty_string(layer_name, "layer name")
        if not isinstance(layer, dict):
            raise AssertionError(f"architecture_contract.yaml: layer {layer_name} must be a mapping")
        patterns = layer.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            raise AssertionError(f"architecture_contract.yaml: layer {layer_name} patterns must be non-empty")
        for pattern in patterns:
            assert_non_empty_string(pattern, f"layer {layer_name} pattern")
        for list_key in ("allowed_imports", "forbidden_imports"):
            values = layer.get(list_key, [])
            if not isinstance(values, list):
                raise AssertionError(f"architecture_contract.yaml: {layer_name}.{list_key} must be a list")
            for value in values:
                assert_non_empty_string(value, f"{layer_name}.{list_key}")

    rules = data.get("rules")
    if not isinstance(rules, dict):
        raise AssertionError("architecture_contract.yaml: rules must be a mapping")
    unknown_rules = set(rules) - KNOWN_RULES
    if unknown_rules:
        raise AssertionError(f"architecture_contract.yaml: unknown rules: {sorted(unknown_rules)}")
    for key, value in rules.items():
        if not isinstance(value, bool):
            raise AssertionError(f"architecture_contract.yaml: rule {key} must be true/false")

    for key in ("allowed_imports", "forbidden_imports", "composition_roots", "runtime_wiring_imports"):
        values = data.get(key)
        if not isinstance(values, list):
            raise AssertionError(f"architecture_contract.yaml: {key} must be a list")
        for value in values:
            assert_non_empty_string(value, key)

    validate_forbidden_source_strings(data.get("forbidden_source_strings"))
    validate_public_api_contracts(data.get("public_api_contracts"))


def validate_forbidden_source_strings(value: Any) -> None:
    if not isinstance(value, list):
        raise AssertionError("architecture_contract.yaml: forbidden_source_strings must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise AssertionError(f"forbidden_source_strings[{index}] must be a mapping")
        scope = item.get("scope")
        strings = item.get("strings")
        if not isinstance(scope, list) or not scope:
            raise AssertionError(f"forbidden_source_strings[{index}].scope must be a non-empty list")
        if not isinstance(strings, list) or not strings:
            raise AssertionError(f"forbidden_source_strings[{index}].strings must be a non-empty list")
        for pattern in scope:
            assert_non_empty_string(pattern, f"forbidden_source_strings[{index}].scope")
        for string in strings:
            assert_non_empty_string(string, f"forbidden_source_strings[{index}].strings")


def validate_public_api_contracts(value: Any) -> None:
    if not isinstance(value, list):
        raise AssertionError("architecture_contract.yaml: public_api_contracts must be a list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise AssertionError(f"public_api_contracts[{index}] must be a mapping")
        module = item.get("module")
        mode = item.get("mode", "minimum")
        expected = item.get("expected", [])
        assert_non_empty_string(module, f"public_api_contracts[{index}].module")
        if mode not in {"minimum", "exact"}:
            raise AssertionError(f"public_api_contracts[{index}].mode must be minimum or exact")
        if not isinstance(expected, list):
            raise AssertionError(f"public_api_contracts[{index}].expected must be a list")
        for name in expected:
            assert_non_empty_string(name, f"public_api_contracts[{index}].expected")


def assert_non_empty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise AssertionError(f"architecture_contract.yaml: {label} must contain non-empty strings")
