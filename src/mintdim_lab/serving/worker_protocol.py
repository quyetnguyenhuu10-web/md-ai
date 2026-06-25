"""JSON event/command protocol for the chat worker stdio channel.

Events emitted (worker → UI):
    loading      — runtime is being initialised
    warming-up   — warmup prompts are running
    ready        — accepting prompt commands
    token        — single decoded token text (one per generate step)
    done         — current prompt generation finished
    error        — fatal or per-prompt error message

Commands accepted (UI → worker):
    prompt       — generate a response for `text`
    shutdown     — terminate the prompt loop and exit cleanly

Importing this module redirects `sys.stdout` to `sys.stderr` so any
incidental prints from downstream libraries do not pollute the JSON
channel. The captured original stdout is forced to UTF-8 and is the
only writer used by `write_event` / `write_error`.
"""

from __future__ import annotations

import json
import sys
from typing import Literal, TypedDict

_real_stdout = sys.stdout
if hasattr(_real_stdout, "reconfigure"):
    try:
        _real_stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
sys.stdout = sys.stderr

EventName = Literal["loading", "warming-up", "ready", "token", "done", "error"]
CommandType = Literal["prompt", "shutdown"]


class Event(TypedDict, total=False):
    event: EventName
    text: str
    message: str
    checkpoint: str
    vocab: str


class Command(TypedDict, total=False):
    type: CommandType
    text: str


def write_event(event: Event) -> None:
    _real_stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    _real_stdout.flush()


def write_error(message: str) -> None:
    write_event({"event": "error", "message": message})


__all__ = [
    "Command",
    "CommandType",
    "Event",
    "EventName",
    "write_error",
    "write_event",
]
