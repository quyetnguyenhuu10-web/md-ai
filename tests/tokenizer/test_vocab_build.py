from __future__ import annotations

from mintdim_lab.cli.commands.train_tokenizer import build_parser
from mintdim_lab.tokenizer.train_byte_bpe import count_jsonl_records, extract_text


def test_extract_text_reads_target_and_appends_multiple_inputs(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    out = tmp_path / "corpus.txt"

    first.write_text('{"target": "statement only"}\n', encoding="utf-8")
    second.write_text(
        '{"prompt": "question", "target": "proof"}\n',
        encoding="utf-8",
    )

    assert extract_text(first, out) == 1
    assert extract_text(second, out) == 2
    assert count_jsonl_records(first) == 1
    assert count_jsonl_records(second) == 1

    assert out.read_text(encoding="utf-8").splitlines() == [
        "statement only",
        "question",
        "proof",
    ]


def test_vocab_build_defaults_to_small_task_vocab_size():
    args = build_parser().parse_args(["--input", "data.jsonl"])

    assert args.vocab_size == 512
    assert args.output_dir.as_posix() == "data_store/tokenizers/byte_bpe_512"
