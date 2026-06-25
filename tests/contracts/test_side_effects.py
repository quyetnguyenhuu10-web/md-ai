from __future__ import annotations

import builtins
import importlib
import socket
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from arch_contract.config import load_contract_config


class ImportSideEffectError(RuntimeError):
    pass


def test_imports_do_not_perform_io_or_spawn_processes(monkeypatch: pytest.MonkeyPatch):
    config = load_contract_config()
    modules = list(config.data.get("side_effect_import_modules", []))
    if not modules:
        pytest.skip("No side_effect_import_modules configured")

    def blocked(*args, **kwargs):
        raise ImportSideEffectError("import attempted IO/process/network side effect")

    monkeypatch.setattr(builtins, "open", blocked)
    monkeypatch.setattr(Path, "open", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    monkeypatch.setattr(socket, "socket", blocked)
    patch_optional_network_clients(monkeypatch, blocked)

    failures: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportSideEffectError as error:
            failures.append(f"{module_name}: {error}")

    assert not failures, "\n".join(failures)


def patch_optional_network_clients(monkeypatch: pytest.MonkeyPatch, blocked) -> None:
    for module_name, attrs in {
        "requests": ["request", "get", "post", "put", "delete", "Session"],
        "httpx": ["request", "get", "post", "put", "delete", "Client", "AsyncClient"],
        "openai": ["OpenAI", "AsyncOpenAI"],
    }.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        assert isinstance(module, ModuleType)
        for attr in attrs:
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, blocked)
