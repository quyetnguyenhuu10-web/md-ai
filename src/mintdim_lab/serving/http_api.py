"""HTTP inference server wrapping ``mintdim_lab.inference.text_generator``.

Run:
    pip install -e .[serve]
    python src/mintdim_lab/cli/main.py serve \
        --checkpoint runs/train/linear_equation_tiny_cpu/checkpoints/step_00000600 \
        --vocab-path data_store/tokenizers/byte_bpe_512/tokenizer.json

Stream tokens via SSE:
    curl -N -X POST http://localhost:8000/generate \\
        -H 'Content-Type: application/json' \\
        -d '{"prompt": "hello", "max_new_tokens": 64}'

Single-process, single-tenant. RNG state is shared across requests so
two concurrent calls will interleave RNG draws — the server is intended
for development and small deployments, not multi-user serving.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import Any

import yaml

from mintdim_lab.inference.model_loader import load_runtime
from mintdim_lab.serving.prompting import chat_prompt_for_model
from mintdim_lab.system.env import force_utf8_stdio
from mintdim_lab.system.paths import CHAT_CHECKPOINT_LIST_PATH, resolve_repo_path

_SENTINEL = object()


try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover — fastapi/pydantic optional install
    BaseModel = object  # type: ignore[assignment]


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None


def normalize_server_args(args: argparse.Namespace) -> argparse.Namespace:
    args.checkpoint_dir = None
    args.preset = None
    args.no_print_decoded_prompt = True
    return args


def _resolve_checkpoint(args: argparse.Namespace) -> None:
    if args.checkpoint and args.vocab_path:
        return
    default_ckpt, default_vocab = _default_checkpoint()
    args.checkpoint = args.checkpoint or default_ckpt
    args.vocab_path = args.vocab_path or default_vocab
    if args.checkpoint is None or args.vocab_path is None:
        raise SystemExit(
            "Need --checkpoint and --vocab-path (or entries in recipes/chat/checkpoints.yaml)."
        )
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.vocab_path.exists():
        raise SystemExit(f"Vocab not found: {args.vocab_path}")


def _default_checkpoint() -> tuple[Path | None, Path | None]:
    if not CHAT_CHECKPOINT_LIST_PATH.exists():
        return None, None

    data = yaml.safe_load(CHAT_CHECKPOINT_LIST_PATH.read_text(encoding="utf-8")) or {}
    for value in data.values():
        raw_paths = value.get("checkpoint", [])
        if not raw_paths or raw_paths[0] is None:
            continue

        ckpt = resolve_repo_path(raw_paths[0])
        if not ckpt.exists():
            continue

        raw_vocabs = value.get("vocab", [])
        vocab = resolve_repo_path(raw_vocabs[0]) if raw_vocabs and raw_vocabs[0] else None
        return ckpt, vocab

    return None, None


def create_app(args: argparse.Namespace) -> Any:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    print(f"Loading runtime from {args.checkpoint}", flush=True)
    model, variables, config, tokenizer, generate_fn, rng = load_runtime(
        args, vocab_path=args.vocab_path
    )

    if not args.no_warmup:
        print("Warming up generate()…", flush=True)
        for prompt in ("warmup ngắn.", "warmup dài hơn một chút để bucket prefill nhỉnh hơn."):
            model_prompt = chat_prompt_for_model(prompt, raw_prompt=bool(args.raw_prompt))
            _, rng = generate_fn(
                model=model,
                variables=variables,
                config=config,
                tokenizer=tokenizer,
                prompt=model_prompt,
                max_new_tokens=2,
                temperature=0.0,
                top_k=0,
                rng=rng,
            )

    state: dict[str, Any] = {"rng": rng}
    app = FastAPI(title="MintDim Inference", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        return {
            "checkpoint": str(args.checkpoint),
            "vocab_path": str(args.vocab_path),
            "vocab_size": int(tokenizer.vocab_size),
            "max_seq_len": int(config.max_seq_len),
            "num_layers": int(config.num_layers),
            "embed_dim": int(config.embed_dim),
        }

    @app.post("/generate")
    async def generate_endpoint(req: GenerateRequest) -> StreamingResponse:
        model_prompt = chat_prompt_for_model(req.prompt, raw_prompt=bool(args.raw_prompt))
        max_new_tokens = (
            req.max_new_tokens if req.max_new_tokens is not None else args.max_new_tokens
        )
        temperature = req.temperature if req.temperature is not None else args.temperature
        top_k = req.top_k if req.top_k is not None else args.top_k

        q: queue.Queue[Any] = queue.Queue()

        def on_token(text: str) -> None:
            q.put(text)

        def run() -> None:
            try:
                _, new_rng = generate_fn(
                    model=model,
                    variables=variables,
                    config=config,
                    tokenizer=tokenizer,
                    prompt=model_prompt,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=int(top_k),
                    rng=state["rng"],
                    on_token=on_token,
                )
                state["rng"] = new_rng
            except Exception as exc:  # noqa: BLE001
                q.put({"error": str(exc)})
            finally:
                q.put(_SENTINEL)

        threading.Thread(target=run, daemon=True).start()

        async def event_stream() -> Any:
            loop = asyncio.get_event_loop()
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is _SENTINEL:
                    yield "event: done\ndata: {}\n\n"
                    return
                if isinstance(item, dict) and "error" in item:
                    yield f"event: error\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                    return
                yield f"event: token\ndata: {json.dumps({'text': item}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def run_server(args: argparse.Namespace) -> None:
    force_utf8_stdio()
    args = normalize_server_args(args)
    _resolve_checkpoint(args)

    try:
        import uvicorn
    except ImportError:
        raise RuntimeError("FastAPI/uvicorn not installed. Install with: pip install -e .[serve]")

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


__all__ = ["create_app", "normalize_server_args", "run_server"]
