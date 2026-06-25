from __future__ import annotations

import argparse


def test_production_cli_main_exposes_dispatcher_boundary():
    import mintdim_lab.cli.main as cli_main

    parser = cli_main._parser()
    action = parser._actions[1]

    assert callable(cli_main.main)
    assert callable(cli_main._parser)
    assert parser.prog == "python src/mintdim_lab/cli/main.py"
    assert tuple(action.choices) == (
        "build-units",
        "train",
        "train-tokenizer",
        "eval",
        "params",
        "chat",
        "serve",
    )


def test_production_cli_commands_expose_command_boundaries():
    import mintdim_lab.cli.commands.build_units as cli_build_units
    import mintdim_lab.cli.commands.chat as cli_chat
    import mintdim_lab.cli.commands.evaluate as cli_evaluate
    import mintdim_lab.cli.commands.params as cli_params
    import mintdim_lab.cli.commands.serve as cli_serve
    import mintdim_lab.cli.commands.train as cli_train
    import mintdim_lab.cli.commands.train_tokenizer as cli_train_tokenizer

    assert callable(cli_build_units.main)
    assert callable(cli_build_units.build_parser)
    assert callable(cli_chat.main)
    assert callable(cli_evaluate.main)
    assert callable(cli_evaluate.build_parser)
    assert callable(cli_chat.build_parser)
    assert callable(cli_params.main)
    assert callable(cli_params.build_parser)
    assert callable(cli_params.run_params_command)
    assert callable(cli_serve.main)
    assert callable(cli_serve.build_parser)
    assert callable(cli_train.main)
    assert callable(cli_train.build_parser)
    assert callable(cli_train.run_train_command)
    assert callable(cli_train_tokenizer.main)


def test_project_uses_cli_package_entrypoint_instead_of_console_script():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
    cli_main = (repo / "src" / "mintdim_lab" / "cli" / "main.py").read_text(
        encoding="utf-8"
    )
    legacy_target = "md" + ".cli.main:main"

    assert "[project.scripts]" not in pyproject
    assert legacy_target not in pyproject
    assert "train-tokenizer" in cli_main
    assert "return train_tokenizer.main(rest)" in cli_main
    assert "return serve.main(rest)" in cli_main
    assert not (repo / "cli.py").exists()
    assert not (repo / "src" / "cli.py").exists()


def test_build_units_cli_reads_unit_build_yaml_from_corpus_dir(monkeypatch, tmp_path):
    import mintdim_lab.cli.commands.build_units as cli_build_units

    calls = []
    corpus_dir = tmp_path / "linear_equation_unit96"
    corpus_dir.mkdir()
    (corpus_dir / "unit_build.yaml").write_text("source: {}\n", encoding="utf-8")

    def run_unit_build_config_yaml(path):
        calls.append(path)
        return {"outputs": []}

    monkeypatch.setattr(
        cli_build_units,
        "run_unit_build_config_yaml",
        run_unit_build_config_yaml,
    )

    ns = argparse.Namespace(
        unit_build_config=str(corpus_dir),
    )

    result = cli_build_units.run_build_units_command(ns)

    assert result["status"] == "ok"
    assert result["unit_build_input"] == str(corpus_dir)
    assert result["unit_build_config"] == str(corpus_dir / "unit_build.yaml")
    assert result["result"] == {"outputs": []}
    assert calls == [corpus_dir / "unit_build.yaml"]


def test_build_units_cli_accepts_explicit_yaml_file(monkeypatch, tmp_path):
    import mintdim_lab.cli.commands.build_units as cli_build_units

    calls = []
    explicit_file = tmp_path / "custom_build_recipe.yaml"
    explicit_file.write_text("source: {}\n", encoding="utf-8")

    def run_unit_build_config_yaml(path):
        calls.append(path)
        return {"outputs": []}

    monkeypatch.setattr(
        cli_build_units,
        "run_unit_build_config_yaml",
        run_unit_build_config_yaml,
    )

    result = cli_build_units.run_build_units_command(
        argparse.Namespace(unit_build_config=str(explicit_file))
    )

    assert result["status"] == "ok"
    assert result["unit_build_input"] == str(explicit_file)
    assert result["unit_build_config"] == str(explicit_file)
    assert calls == [explicit_file]
