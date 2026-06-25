from __future__ import annotations

import io

from mintdim_lab.trainer.terminal_ui import TerminalTrainUI


class _NonSeekableStream:
    def __init__(self) -> None:
        self.buffer = io.StringIO()

    def write(self, text: str) -> int:
        return self.buffer.write(text)

    def flush(self) -> None:
        return None

    def tell(self) -> int:
        raise OSError("not seekable")

    def getvalue(self) -> str:
        return self.buffer.getvalue()


def test_terminal_ui_keeps_one_live_progress_and_logs_text_only() -> None:
    stream = io.StringIO()
    ui = TerminalTrainUI(
        enabled=True,
        total_steps=10,
        log_every=2,
        stream=stream,
        bar_width=10,
    )

    ui.step({"step": 1, "loss_mean": 2.0, "token_count": 8})
    ui.step({"step": 2, "loss_mean": 1.5, "token_count": 8})

    text = stream.getvalue()

    assert "\rtrain [" in text
    assert "step 2  loss 1.5000  tok 8\n" in text
    assert "train [" in text
    assert "2/10" in text
    assert "20.00%" in text
    assert "[##--------]" in text

    log_line = [line for line in text.split("\n") if "step 2  loss 1.5000  tok 8" in line][0]
    assert "[" not in log_line
    assert "]" not in log_line


def test_terminal_ui_redraws_without_clearing_the_line_until_text_log() -> None:
    stream = _NonSeekableStream()
    ui = TerminalTrainUI(
        enabled=True,
        total_steps=10,
        log_every=3,
        stream=stream,
        bar_width=10,
    )

    ui.step({"step": 1, "loss_mean": 2.0, "token_count": 8})
    ui.step({"step": 2, "loss_mean": 1.5, "token_count": 8})

    assert "\033[2K" not in stream.getvalue()

    ui.step({"step": 3, "loss_mean": 1.25, "token_count": 8})

    assert stream.getvalue().count("\033[2K") == 1
    assert "[###-------]" in stream.getvalue()


def test_terminal_ui_heartbeat_moves_while_waiting_for_next_step() -> None:
    stream = io.StringIO()
    ui = TerminalTrainUI(
        enabled=True,
        total_steps=10,
        log_every=100,
        stream=stream,
        bar_width=10,
    )

    ui.start()
    first = stream.getvalue()
    ui._animation_tick()
    second = stream.getvalue()
    ui.stop()

    assert "[>---------]" in first
    assert "[->--------]" in second
    assert ui._thread is None


def test_terminal_ui_disabled_writes_nothing() -> None:
    stream = io.StringIO()
    ui = TerminalTrainUI(
        enabled=False,
        total_steps=10,
        log_every=1,
        stream=stream,
    )

    ui.step({"step": 1, "loss_mean": 2.0, "token_count": 8})
    ui.checkpoint(1)
    ui.finish(
        {
            "final_step": 1,
            "checkpoint": {"saved_steps": [1]},
            "last_metrics": {"step": 1, "loss_mean": 2.0, "token_count": 8},
        }
    )

    assert stream.getvalue() == ""


def test_terminal_ui_checkpoint_replaces_progress_then_redraws_one_progress() -> None:
    stream = io.StringIO()
    ui = TerminalTrainUI(
        enabled=True,
        total_steps=10,
        log_every=100,
        stream=stream,
        bar_width=10,
    )

    ui.step({"step": 4, "loss_mean": 1.25, "token_count": 8})
    ui.checkpoint(4)

    text = stream.getvalue()

    assert "ckpt  saved step_00000004\n" in text
    assert text.rstrip().endswith("4/10  40.00%")


def test_terminal_ui_finish_replaces_progress_with_done_text() -> None:
    stream = io.StringIO()
    ui = TerminalTrainUI(
        enabled=True,
        total_steps=10,
        log_every=100,
        stream=stream,
        bar_width=10,
    )

    ui.step({"step": 10, "loss_mean": 1.0, "token_count": 8})
    ui.finish(
        {
            "final_step": 10,
            "checkpoint": {"saved_steps": [5, 10]},
            "last_metrics": {"step": 10, "loss_mean": 1.0, "token_count": 8},
        }
    )

    text = stream.getvalue()

    assert text.endswith("done  step 10/10  loss 1.0000  saved 2 ckpt\n")


def test_terminal_ui_parser_accepts_ui_flags() -> None:
    from mintdim_lab.cli.commands.train import build_parser

    ns = build_parser().parse_args(
        [
            "--no-ui",
            "--log-every",
            "25",
            "--progress-width",
            "32",
        ]
    )

    assert ns.ui is False
    assert ns.log_every == 25
    assert ns.progress_width == 32
