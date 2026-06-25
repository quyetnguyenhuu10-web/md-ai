"""Dependency-free terminal progress UI for training.

This module owns presentation only. It does not read data, select runtime
devices, compile JAX functions, save checkpoints, or mutate training state.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from typing import Any, TextIO

_CLEAR_LINE = "\033[2K"
_ANIMATION_INTERVAL_SEC = 0.25


@dataclass
class TerminalTrainUI:
    """Render one live progress bar plus text log lines.

    The active progress bar is never committed as historical output. When a log
    event happens, the current progress row is cleared and replaced with text,
    then one new progress row is drawn below it.
    """

    total_steps: int
    enabled: bool = True
    log_every: int = 100
    stream: TextIO = sys.stderr
    bar_width: int = 28
    _last_line_len: int = field(default=0, init=False)
    _last_metrics: dict[str, Any] | None = field(default=None, init=False)
    _active_line_pos: int | None = field(default=None, init=False)
    _pulse_index: int = field(default=0, init=False)
    _running: bool = field(default=False, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Start a lightweight heartbeat so slow steps still look alive."""
        if not self.enabled:
            return

        with self._lock:
            if self._running:
                return

            self._running = True
            self._stop_event.clear()
            self._rewrite_progress(self._last_metrics or {"step": 0})
            self._thread = threading.Thread(
                target=self._animate,
                name="mintdim-train-ui",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the heartbeat worker without committing the live row."""
        if not self.enabled:
            return

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

        with self._lock:
            self._running = False
            self._thread = None

    def step(self, metrics: dict[str, Any]) -> None:
        """Render a completed training step."""
        if not self.enabled:
            return

        with self._lock:
            self._last_metrics = dict(metrics)
            step = int(metrics["step"])

            if int(self.log_every) > 0 and step % int(self.log_every) == 0:
                self._replace_active_line_with_text(self._log_line(metrics))

            self._rewrite_progress(metrics)

    def checkpoint(self, step: int) -> None:
        """Render a checkpoint save event as text, then redraw progress."""
        if not self.enabled:
            return

        with self._lock:
            self._replace_active_line_with_text(f"ckpt  saved step_{int(step):08d}")

            if self._last_metrics is not None:
                self._rewrite_progress(self._last_metrics)

    def finish(self, result: dict[str, Any]) -> None:
        """Replace the active progress row with the final text summary."""
        if not self.enabled:
            return

        self.stop()

        with self._lock:
            last_metrics = result.get("last_metrics") or self._last_metrics or {}
            final_step = int(result.get("final_step", last_metrics.get("step", 0)))
            saved_steps = result.get("checkpoint", {}).get("saved_steps", ())
            loss = _format_number(last_metrics.get("loss_mean"))

            self._replace_active_line_with_text(
                f"done  step {final_step}/{self.total_steps}  "
                f"loss {loss}  saved {len(saved_steps)} ckpt"
            )

    def _animate(self) -> None:
        while not self._stop_event.wait(_ANIMATION_INTERVAL_SEC):
            self._animation_tick()

    def _animation_tick(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._pulse_index += 1
            self._rewrite_progress(self._last_metrics or {"step": 0})

    def _rewrite_progress(self, metrics: dict[str, Any]) -> None:
        line = self._progress_line(metrics)
        clear_tail = " " * max(0, self._last_line_len - len(line))
        self._discard_seekable_active_line()
        self._active_line_pos = self._stream_position()
        self.stream.write("\r" + line + clear_tail)
        self.stream.flush()
        self._last_line_len = len(line)

    def _replace_active_line_with_text(self, text: str) -> None:
        discarded = self._discard_seekable_active_line()
        prefix = "" if discarded else "\r" + _CLEAR_LINE
        self.stream.write(prefix + text + "\n")
        self.stream.flush()
        self._last_line_len = 0

    def _discard_seekable_active_line(self) -> bool:
        if self._active_line_pos is None:
            return False

        try:
            self.stream.seek(self._active_line_pos)
            self.stream.truncate()
        except (AttributeError, OSError):
            self._active_line_pos = None
            return False

        self._active_line_pos = None
        return True

    def _stream_position(self) -> int | None:
        try:
            return int(self.stream.tell())
        except (AttributeError, OSError):
            return None

    def _progress_line(self, metrics: dict[str, Any]) -> str:
        step = int(metrics["step"])
        total = max(1, int(self.total_steps))
        ratio = min(1.0, max(0.0, step / total))
        percent = ratio * 100.0
        bar = _bar(
            ratio,
            width=int(self.bar_width),
            pulse_index=self._pulse_index,
            active=self._running and step < total,
        )
        step_width = len(str(total))

        return f"train [{bar}] {step:0{step_width}d}/{total} {percent:6.2f}%"

    def _log_line(self, metrics: dict[str, Any]) -> str:
        step = int(metrics["step"])
        loss = _format_number(metrics.get("loss_mean"))
        token_count = _format_number(metrics.get("token_count"), digits=0)
        effective_batch = metrics.get("effective_batch_size")
        if effective_batch is None:
            return f"step {step}  loss {loss}  tok {token_count}"
        return (
            f"step {step}  loss {loss}  tok {token_count}  "
            f"eff_b {_format_number(effective_batch, digits=0)}"
        )


def _bar(
    ratio: float,
    *,
    width: int,
    pulse_index: int = 0,
    active: bool = False,
) -> str:
    resolved_width = max(1, int(width))
    filled = int(ratio * resolved_width)
    filled = min(resolved_width, max(0, filled))
    remaining = resolved_width - filled
    if not active or remaining <= 0:
        return "#" * filled + "-" * remaining

    marker = int(pulse_index) % remaining
    return "#" * filled + "-" * marker + ">" + "-" * (remaining - marker - 1)


def _format_number(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if digits == 0:
        return str(int(number))

    return f"{number:.{digits}f}"
