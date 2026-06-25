from __future__ import annotations

import ast
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from .config import ContractConfig
from .files import find_layer_for_path, is_composition_root, relpath
from .imports import ImportRecord, build_python_module_index, import_display, is_stdlib_module


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    line: int
    detail: str

    @property
    def key(self) -> str:
        return f"{self.rule}|{self.path}|{self.line}|{self.detail}"

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail}"


def assert_no_unbaselined_violations(violations: list[Violation], config: ContractConfig) -> None:
    unbaselined = [violation for violation in violations if violation.key not in config.baseline]
    if unbaselined:
        rendered = "\n".join(violation.render() for violation in unbaselined[:80])
        extra = "" if len(unbaselined) <= 80 else f"\n... {len(unbaselined) - 80} more"
        raise AssertionError(rendered + extra)


def layer_for_module(module: str, config: ContractConfig) -> str | None:
    modules = build_python_module_index(config)
    info = modules.get(module)
    if info is None:
        return None
    return find_layer_for_path(info.path, config)


def layer_import_boundary_violations(config: ContractConfig, imports: list[ImportRecord]) -> list[Violation]:
    violations: list[Violation] = []
    module_layers = build_module_layer_map(config)
    for record in imports:
        source_layer = find_layer_for_path(record.path, config)
        if source_layer is None:
            continue
        layer = config.layers[source_layer]
        imported_layer = module_layers.get(record.resolved_module) if record.resolved_module else None
        imported_module = record.resolved_module or record.imported_module

        forbidden = list(config.data.get("forbidden_imports", [])) + list(layer.get("forbidden_imports", []))
        for forbidden_pattern in forbidden:
            if import_matches(forbidden_pattern, imported_module, imported_layer):
                violations.append(
                    Violation(
                        rule="layer_forbidden_import",
                        path=relpath(record.path, config.repo_root),
                        line=record.line,
                        detail=(
                            f"layer={source_layer} import={import_display(record)} "
                            f"matched_forbidden={forbidden_pattern}"
                        ),
                    )
                )

        allowed = list(config.data.get("allowed_imports", [])) + list(layer.get("allowed_imports", []))
        if not is_import_allowed(record, imported_layer, allowed):
            violations.append(
                Violation(
                    rule="layer_import_not_allowed",
                    path=relpath(record.path, config.repo_root),
                    line=record.line,
                    detail=(
                        f"layer={source_layer} import={import_display(record)} "
                        f"resolved={record.resolved_module or '<external>'}"
                    ),
                )
            )
    return violations


def build_module_layer_map(config: ContractConfig) -> dict[str, str]:
    modules = build_python_module_index(config)
    layers: dict[str, str] = {}
    for module, info in modules.items():
        layer = find_layer_for_path(info.path, config)
        if layer is not None:
            layers[module] = layer
    return layers


def is_import_allowed(record: ImportRecord, imported_layer: str | None, allowed: list[str]) -> bool:
    imported_module = record.resolved_module or record.imported_module
    if imported_layer and imported_layer in allowed:
        return True
    if record.resolved_module is None:
        if "stdlib" in allowed and is_stdlib_module(imported_module):
            return True
        if "external" in allowed and not is_stdlib_module(imported_module):
            return True
    return any(import_matches(pattern, imported_module, imported_layer) for pattern in allowed)


def import_matches(pattern: str, imported_module: str, imported_layer: str | None) -> bool:
    if imported_layer and pattern == imported_layer:
        return True
    normalized = imported_module.replace("/", ".")
    return fnmatch(normalized, pattern) or normalized == pattern or normalized.startswith(f"{pattern}.")


def private_import_violations(config: ContractConfig, imports: list[ImportRecord]) -> list[Violation]:
    allow_tests = config.rules.get("allow_private_imports_for_tests", False)
    violations: list[Violation] = []
    for record in imports:
        if not record.imported_name or not record.imported_name.startswith("_"):
            continue
        if record.imported_name.startswith("__"):
            continue
        relative = relpath(record.path, config.repo_root)
        if allow_tests and relative.startswith("tests/"):
            continue
        if is_same_package_private_import(record):
            continue
        violations.append(
            Violation(
                rule="private_cross_import",
                path=relative,
                line=record.line,
                detail=(
                    f"module={record.imported_module or '<relative>'} "
                    f"private_name={record.imported_name}"
                ),
            )
        )
    return violations


def is_same_package_private_import(record: ImportRecord) -> bool:
    if record.resolved_module is None:
        return False
    source_package = record.module_name.rsplit(".", 1)[0]
    imported_package = record.resolved_module.rsplit(".", 1)[0]
    return source_package == imported_package


def relative_parent_import_violations(config: ContractConfig, imports: list[ImportRecord]) -> list[Violation]:
    if not config.rules.get("forbid_relative_parent_imports", False):
        return []
    violations: list[Violation] = []
    for record in imports:
        if record.level <= 1:
            continue
        violations.append(
            Violation(
                rule="relative_parent_import",
                path=relpath(record.path, config.repo_root),
                line=record.line,
                detail=f"import={import_display(record)} level={record.level}",
            )
        )
    return violations


def runtime_wiring_violations(config: ContractConfig, imports: list[ImportRecord]) -> list[Violation]:
    if not config.rules.get("forbid_runtime_wiring_outside_composition_root", False):
        return []
    runtime_names = list(config.data.get("runtime_wiring_imports", []))
    violations: list[Violation] = []
    for record in imports:
        if is_composition_root(record.path, config):
            continue
        imported = record.resolved_module or record.imported_module
        matched = next((name for name in runtime_names if import_matches(name, imported, None)), None)
        if matched:
            violations.append(
                Violation(
                    rule="runtime_wiring_import_outside_composition_root",
                    path=relpath(record.path, config.repo_root),
                    line=record.line,
                    detail=f"import={import_display(record)} matched={matched}",
                )
            )
    violations.extend(runtime_call_violations(config, runtime_names))
    return violations


def runtime_call_violations(config: ContractConfig, runtime_names: list[str]) -> list[Violation]:
    from .files import iter_python_files
    from .imports import parse_python_ast

    call_names = {name for name in runtime_names if "." in name}
    violations: list[Violation] = []
    for path in iter_python_files(config):
        if is_composition_root(path, config):
            continue
        tree = parse_python_ast(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = dotted_call_name(node.func)
            if not name:
                continue
            matched = next((runtime for runtime in call_names if name == runtime or name.endswith(f".{runtime}")), None)
            if matched:
                violations.append(
                    Violation(
                        rule="runtime_wiring_call_outside_composition_root",
                        path=relpath(path, config.repo_root),
                        line=node.lineno,
                        detail=f"call={name} matched={matched}",
                    )
                )
    return violations


def dotted_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def forbidden_source_string_violations(config: ContractConfig) -> list[Violation]:
    from .files import iter_source_files, path_matches

    violations: list[Violation] = []
    files = iter_source_files(config)
    for item in config.data.get("forbidden_source_strings", []):
        scopes = list(item.get("scope", []))
        strings = list(item.get("strings", []))
        for path in files:
            relative = relpath(path, config.repo_root)
            if not path_matches(relative, scopes):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for forbidden in strings:
                if forbidden in text:
                    line = text[: text.index(forbidden)].count("\n") + 1
                    violations.append(
                        Violation(
                            rule="forbidden_source_string",
                            path=relative,
                            line=line,
                            detail=f"string={forbidden!r}",
                        )
                    )
    return violations


def parse_static_all(path: Path) -> list[str] | None:
    from .imports import parse_python_ast

    tree = parse_python_ast(path)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            values: list[str] = []
            for item in node.value.elts:
                if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                    return []
                values.append(item.value)
            return values
        return []
    return None
