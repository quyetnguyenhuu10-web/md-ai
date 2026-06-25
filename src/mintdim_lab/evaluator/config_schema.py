from __future__ import annotations

SECTION_FIELDS = {
    "checkpoint": {
        "path": "checkpoint",
    },
    "data": {
        "eval_path": "eval_path",
    },
    "template": {
        "sequence": "template_sequence",
        "input_until": "template_input_until",
        "target": "template_target",
    },
    "vocab": {
        "path": "vocab_path",
    },
    "runtime": {
        "batch_size": "batch_size",
        "max_new_tokens": "max_new_tokens",
        "fixed_seq_len": "fixed_seq_len",
        "limit": "limit",
        "stop_at_target_length": "stop_at_target_length",
    },
    "scoring": {
        "ignore_whitespace": "ignore_whitespace",
        "normalize_line_endings": "normalize_line_endings",
        "case_insensitive": "case_insensitive",
    },
    "output": {
        "json_path": "json_output",
        "wrong_limit": "wrong_limit",
    },
}

REQUIRED_CONFIG_FIELDS = (
    "checkpoint",
    "eval_path",
    "vocab_path",
    "template_sequence",
    "template_input_until",
    "template_target",
    "batch_size",
    "max_new_tokens",
    "limit",
    "ignore_whitespace",
    "normalize_line_endings",
    "case_insensitive",
    "wrong_limit",
)

__all__ = ["REQUIRED_CONFIG_FIELDS", "SECTION_FIELDS"]
