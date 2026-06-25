"""Build a byte-level BPE vocabulary from a JSONL text corpus."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from .audit import TEXT_FIELDS, iter_jsonl_text, unicode_codepoints, write_unk_report
from .rules import REQUIRED_VOCAB_TOKENS, TOKENIZER_SPECIAL_TOKENS


def count_jsonl_records(input_path: Path) -> int:
    n_records = 0
    with input_path.open(encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                n_records += 1
    return n_records


def extract_text(input_path: Path, out_path: Path) -> int:
    n_lines = 0
    with out_path.open("a", encoding="utf-8") as f_out:
        for value in iter_jsonl_text(input_path):
            f_out.write(value + "\n")
            n_lines += 1
    return n_lines


def train_byte_bpe_from_args(args: Any) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path) for path in args.input]
    for input_path in input_paths:
        if not input_path.exists():
            raise SystemExit(f"Input not found: {input_path}")

    special_tokens = list(TOKENIZER_SPECIAL_TOKENS)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        n_records = 0
        n_segments = 0
        for input_path in input_paths:
            n_records += count_jsonl_records(input_path)
            n_segments += extract_text(input_path, tmp_path)
        if n_segments == 0:
            raise SystemExit(f"No usable text extracted from {input_paths}")
        print(
            f"Read {n_records} JSONL records; extracted {n_segments} text segments "
            f"from {len(input_paths)} input file(s)"
        )

        byte_alphabet = list(
            dict.fromkeys([*ByteLevelPreTokenizer.alphabet(), *REQUIRED_VOCAB_TOKENS])
        )
        min_vocab = len(byte_alphabet) + len(special_tokens)
        if int(args.vocab_size) < min_vocab:
            raise SystemExit(
                f"vocab_size {args.vocab_size} is too small - need at least "
                f"{min_vocab} ({len(byte_alphabet)} byte tokens + "
                f"{len(special_tokens)} special tokens)."
            )

        tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=int(args.vocab_size),
            min_frequency=int(args.min_frequency),
            special_tokens=special_tokens,
            initial_alphabet=byte_alphabet,
            show_progress=True,
        )

        tokenizer.train([str(tmp_path)], trainer)

        tokenizer.add_special_tokens(special_tokens)
        tokenizer.enable_padding(pad_id=0, pad_token="<pad>")

        missing_tokens = [
            token for token in REQUIRED_VOCAB_TOKENS if tokenizer.token_to_id(token) is None
        ]
        if missing_tokens:
            raise RuntimeError(f"Tokenizer is missing required vocab tokens: {missing_tokens}")

        tokenizer_path = output_dir / "tokenizer.json"
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))

        actual_vocab_size = tokenizer.get_vocab_size()
        print(f"Wrote {tokenizer_path}")
        print(f"vocab_size: {actual_vocab_size}")

        for token in special_tokens:
            tid = tokenizer.token_to_id(token)
            print(f"  {token!r:30}  id={tid}")

        for token in REQUIRED_VOCAB_TOKENS:
            tid = tokenizer.token_to_id(token)
            print(f"  {token!r:30}  id={tid}")

        print(f"  {'<byte_alphabet>':30}  size={len(byte_alphabet)}")

        if args.unk_report:
            report_path = Path(str(args.unk_report))
        else:
            report_path = output_dir / "tokenizer_unk.jsonl"

        print(f"Auditing <unk> pieces into {report_path}", flush=True)
        total_unk, unique_unk = write_unk_report(
            input_paths=input_paths,
            tokenizer=tokenizer,
            report_path=report_path,
            max_items=int(args.unk_report_max),
            log_every=int(args.unk_report_log_every),
        )
        print(f"unk_total: {total_unk}")
        print(f"unk_unique: {unique_unk}")
        print(f"unk_report: {report_path}")
    finally:
        tmp_path.unlink(missing_ok=True)


__all__ = [
    "TEXT_FIELDS",
    "count_jsonl_records",
    "extract_text",
    "train_byte_bpe_from_args",
    "unicode_codepoints",
    "write_unk_report",
]
