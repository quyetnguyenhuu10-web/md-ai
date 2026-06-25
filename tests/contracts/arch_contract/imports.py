from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from sys import stdlib_module_names

from .config import ContractConfig
from .files import iter_python_files, relpath


@dataclass(frozen=True)
class ImportRecord:
    path: Path
    module_name: str
    line: int
    imported_module: str
    imported_name: str | None
    level: int
    kind: str
    resolved_module: str | None


@dataclass(frozen=True)
class ModuleInfo:
    name: str
    path: Path
    is_package: bool


def parse_python_ast(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise AssertionError(f"{path}:{error.lineno}: Python syntax error blocks contract scan: {error}") from error


def build_python_module_index(config: ContractConfig) -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    roots = [config.repo_root / root for root in config.source_roots]
    for path in iter_python_files(config):
        for root in roots:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            parts = list(relative.with_suffix("").parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts = parts[:-1]
            if not parts:
                continue
            module = ".".join(parts)
            modules[module] = ModuleInfo(name=module, path=path, is_package=is_package)
            break
    return modules


def collect_python_imports(config: ContractConfig) -> list[ImportRecord]:
    modules = build_python_module_index(config)
    records: list[ImportRecord] = []
    for path in iter_python_files(config):
        module_info = module_info_for_path(path, modules)
        if module_info is None:
            continue
        tree = parse_python_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_module = alias.name
                    records.append(
                        ImportRecord(
                            path=path,
                            module_name=module_info.name,
                            line=node.lineno,
                            imported_module=imported_module,
                            imported_name=None,
                            level=0,
                            kind="import",
                            resolved_module=resolve_imported_module(imported_module, None, modules),
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                base_module = node.module or ""
                for alias in node.names:
                    absolute_module = resolve_from_module(
                        current=module_info,
                        base_module=base_module,
                        imported_name=alias.name,
                        level=node.level,
                        modules=modules,
                    )
                    records.append(
                        ImportRecord(
                            path=path,
                            module_name=module_info.name,
                            line=node.lineno,
                            imported_module=base_module,
                            imported_name=alias.name,
                            level=node.level,
                            kind="from",
                            resolved_module=absolute_module,
                        )
                    )
    return records


def module_info_for_path(path: Path, modules: dict[str, ModuleInfo]) -> ModuleInfo | None:
    for module in modules.values():
        if module.path == path:
            return module
    return None


def resolve_from_module(
    current: ModuleInfo,
    base_module: str,
    imported_name: str,
    level: int,
    modules: dict[str, ModuleInfo],
) -> str | None:
    if level:
        package_parts = current.name.split(".") if current.is_package else current.name.split(".")[:-1]
        if level > 1:
            package_parts = package_parts[: -(level - 1)]
        if base_module:
            package_parts.extend(base_module.split("."))
        candidate_base = ".".join(part for part in package_parts if part)
    else:
        candidate_base = base_module

    return resolve_imported_module(candidate_base, imported_name, modules)


def resolve_imported_module(
    base_module: str,
    imported_name: str | None,
    modules: dict[str, ModuleInfo],
) -> str | None:
    candidates: list[str] = []
    if imported_name and imported_name != "*":
        candidates.append(f"{base_module}.{imported_name}" if base_module else imported_name)
    if base_module:
        candidates.append(base_module)
    for candidate in candidates:
        if candidate in modules:
            return candidate
        parent = candidate
        while "." in parent:
            parent = parent.rsplit(".", 1)[0]
            if parent in modules:
                return parent
    return None


def is_stdlib_module(module: str) -> bool:
    root = module.split(".", 1)[0]
    return root in stdlib_module_names or root in {"__future__", "builtins"}


def import_display(record: ImportRecord) -> str:
    if record.kind == "import":
        return record.imported_module
    dots = "." * record.level
    name = f" import {record.imported_name}" if record.imported_name else ""
    return f"from {dots}{record.imported_module}{name}"


def format_import_location(record: ImportRecord, config: ContractConfig) -> str:
    return f"{relpath(record.path, config.repo_root)}:{record.line}"
