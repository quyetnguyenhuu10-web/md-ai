from __future__ import annotations

import importlib


def test_mintdim_lab_package_shell_imports():
    modules = [
        "mintdim_lab",
        "mintdim_lab.problems",
        "mintdim_lab.problems.linear_equation",
        "mintdim_lab.corpus",
        "mintdim_lab.tokenizer",
        "mintdim_lab.model",
        "mintdim_lab.model.layers",
        "mintdim_lab.trainer",
        "mintdim_lab.trainer.config",
        "mintdim_lab.trainer.config.training",
        "mintdim_lab.trainer.config.unit_read",
        "mintdim_lab.trainer.gradient_accumulation",
        "mintdim_lab.trainer.loop",
        "mintdim_lab.trainer.objective",
        "mintdim_lab.trainer.optimizer",
        "mintdim_lab.trainer.state",
        "mintdim_lab.trainer.update_step",
        "mintdim_lab.evaluator",
        "mintdim_lab.serving",
        "mintdim_lab.run_setup",
        "mintdim_lab.run_setup.run_context",
        "mintdim_lab.run_setup.run_directory",
        "mintdim_lab.system",
        "mintdim_lab.cli",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name
