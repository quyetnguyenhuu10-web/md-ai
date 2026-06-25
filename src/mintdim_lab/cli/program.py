"""Shared command-line program labels."""

MODULE_PROG = "python src/mintdim_lab/cli/main.py"


def command_prog(command: str) -> str:
    return f"{MODULE_PROG} {command}"


__all__ = ["MODULE_PROG", "command_prog"]
