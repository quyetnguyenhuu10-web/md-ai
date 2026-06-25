from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mintdim_lab.inference.checkpoint_selection import select_checkpoint
from mintdim_lab.inference.text_generator import generate
from mintdim_lab.model import Transformer, TransformerConfig, make_attention_types
from mintdim_lab.model.builder import text_metadata_fields
from mintdim_lab.system.checkpoint_io import (
    checkpoint_logits_head,
    checkpoint_metadata,
    checkpoint_num_embed,
    checkpoint_to_variables,
    load_checkpoint,
)
from mintdim_lab.system.paths import REPO_ROOT
from mintdim_lab.tokenizer import load_tokenizer

MODEL_CONFIG_METADATA_FIELDS: tuple[str, ...] = (
    "num_embed",
    *text_metadata_fields(include_num_embed=False),
)
REBUILDABLE_MODEL_METADATA_FIELDS = MODEL_CONFIG_METADATA_FIELDS

MODEL_CONFIG_METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "max_seq_len": ("max_seq_len", "model_max_seq_len"),
}


def load_runtime(
    args: Any,
    *,
    vocab_path: Path | str | None = None,
) -> tuple[Any, dict[str, Any], Any, Any, Any, Any]:
    checkpoint_path = args.checkpoint or select_checkpoint(args.checkpoint_dir)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    variables = checkpoint_to_variables(checkpoint)
    tokenizer = _load_tokenizer_for_checkpoint(
        checkpoint=checkpoint,
        variables=variables,
        args=args,
        vocab_path=vocab_path,
    )

    config, metadata_overrides = config_from_checkpoint_metadata(checkpoint)
    config = _resolve_num_embed(
        config=config,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        variables=variables,
    )
    config = _resolve_logits_head(
        config=config,
        checkpoint=checkpoint,
        variables=variables,
    )

    model, generate_fn = build_model_and_generate_fn(config=config)
    rng = make_rng(args.seed)

    print("Config source: checkpoint metadata")
    print(f"Config metadata fields loaded: {len(metadata_overrides)}")
    print(
        "QK position: "
        f"local_qk_dim={config.local_qk_dim}, "
        f"local_rope_dim={config.local_rope_dim}, "
        f"local_nope_dim={config.local_qk_dim - config.local_rope_dim}, "
        f"global_qk_dim={config.global_qk_dim}, "
        f"global_rope_dim={config.global_rope_dim}, "
        f"global_nope_dim={config.global_qk_dim - config.global_rope_dim}"
    )
    print(f"Tokenizer: {args.tokenizer}")
    print(f"num_embed: {config.num_embed}")
    print(f"logits_head: {config.logits_head}")
    print(f"ffw_activation: {config.ffw_activation}")

    return model, variables, config, tokenizer, generate_fn, rng


def config_from_checkpoint_metadata(checkpoint: Any) -> tuple[TransformerConfig, dict[str, Any]]:
    metadata = checkpoint_metadata(checkpoint)
    _require_rebuildable_model_metadata(metadata)

    values = _checkpoint_metadata_config_values(metadata, base_config=None)
    required_field_names = {
        field.name
        for field in dataclasses.fields(TransformerConfig)
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    }
    missing = sorted(required_field_names - set(values))
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Checkpoint metadata is incomplete for constructing model config. "
            f"Missing fields: {joined}."
        )

    config = TransformerConfig(
        **{
            field.name: values[field.name]
            for field in dataclasses.fields(TransformerConfig)
            if field.name in values
        }
    )
    config.validate()
    return config, values


def _checkpoint_metadata_config_values(
    metadata: Mapping[str, Any],
    *,
    base_config: Any | None,
) -> dict[str, Any]:
    replacements: dict[str, Any] = {}
    converters = {
        "num_embed": int,
        "max_seq_len": int,
        "num_layers": int,
        "embed_dim": int,
        "dense_hidden_dim": int,
        "ffw_activation": lambda value: str(value).strip().lower(),
        "num_heads": int,
        "num_local_kv_heads": int,
        "num_global_kv_heads": lambda value: None if value is None else int(value),
        "v_head_dim": int,
        "sliding_window_size": int,
        "final_logit_softcap": lambda value: None if value is None else float(value),
        "local_qk_logits_softcap": lambda value: None if value is None else float(value),
        "global_qk_logits_softcap": lambda value: None if value is None else float(value),
        "use_post_attn_norm": _to_bool,
        "use_post_ffw_norm": _to_bool,
        "qk_norm_enabled": _to_bool,
        "qk_norm_with_scale": _to_bool,
        "local_qk_dim": int,
        "global_qk_dim": int,
        "local_rope_dim": int,
        "global_rope_dim": int,
        "local_rope_base": int,
        "global_rope_base": int,
        "local_rope_scale": float,
        "global_rope_scale": float,
        "logits_head": lambda value: str(value).strip().lower(),
        "mtp_num_future_tokens": int,
        "mtp_loss_weight": float,
        "use_gradient_checkpointing": _to_bool,
        "use_fused_attention": _to_bool,
        "use_flash_attention": _to_bool,
    }
    for field, convert in converters.items():
        found, value = _metadata_value(metadata, field)
        if found:
            replacements[field] = convert(value)

    found_old_softcap, old_softcap = _metadata_value(metadata, "attn_logits_softcap")
    if found_old_softcap:
        softcap = None if old_softcap is None else float(old_softcap)
        replacements.setdefault("local_qk_logits_softcap", softcap)
        replacements.setdefault("global_qk_logits_softcap", softcap)

    if "qk_norm_enabled" not in replacements and "qk_norm_with_scale" in replacements:
        replacements["qk_norm_enabled"] = bool(replacements["qk_norm_with_scale"])

    if not _metadata_has(metadata, "ffw_activation"):
        replacements["ffw_activation"] = "gelu"

    found_pattern, pattern = _metadata_value(metadata, "attention_pattern")
    if found_pattern:
        default_num_layers = None if base_config is None else base_config.num_layers
        num_layers = int(replacements.get("num_layers", default_num_layers))
        replacements["attention_types"] = make_attention_types(
            _attention_pattern_items(pattern),
            num_layers=num_layers,
        )

    return replacements


def _require_rebuildable_model_metadata(metadata: Mapping[str, Any]) -> None:
    raw_required = metadata.get("model_config_metadata_fields") or MODEL_CONFIG_METADATA_FIELDS
    required = tuple(str(field) for field in _sequence_items(raw_required))
    unsupported = sorted(set(required) - set(MODEL_CONFIG_METADATA_FIELDS))
    if unsupported:
        joined = ", ".join(unsupported)
        raise SystemExit(
            f"Checkpoint metadata has model config fields this runtime cannot build: {joined}"
        )

    missing = sorted(field for field in required if not _metadata_has(metadata, field))
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Checkpoint metadata is incomplete for rebuilding the chat model config. "
            f"Missing fields: {joined}. "
            "Use a checkpoint saved by the current trainer, or pass a checkpoint whose "
            "metadata contains the full model config."
        )


def _metadata_has(metadata: Mapping[str, Any], field: str) -> bool:
    found, _ = _metadata_value(metadata, field)
    return found


def _metadata_value(metadata: Mapping[str, Any], field: str) -> tuple[bool, Any]:
    for key in MODEL_CONFIG_METADATA_ALIASES.get(field, (field,)):
        if key in metadata:
            return True, metadata[key]
    return False, None


def _attention_pattern_items(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, Sequence):
        return [item.decode("utf-8") if isinstance(item, bytes) else item for item in value]
    if isinstance(value, Mapping):
        return [
            item.decode("utf-8") if isinstance(item, bytes) else item
            for item in _sequence_items(value)
        ]
    raise SystemExit(f"Invalid checkpoint attention_pattern metadata: {value!r}")


def _sequence_items(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        items = list(value.items())
        try:
            items.sort(key=lambda item: int(item[0]))
        except (TypeError, ValueError):
            items.sort(key=lambda item: str(item[0]))
        return [item[1] for item in items]

    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]

    if isinstance(value, bytes):
        return [part.strip() for part in value.decode("utf-8").split(",") if part.strip()]

    if isinstance(value, Sequence):
        return list(value)

    return [value]


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def build_model_and_generate_fn(*, config: Any) -> tuple[Any, Any]:
    return Transformer(config), generate


def make_rng(seed: int) -> Any:
    import jax

    return jax.random.PRNGKey(int(seed))


def _load_tokenizer_for_checkpoint(
    *,
    checkpoint: dict[str, Any],
    variables: dict[str, Any],
    args: Any,
    vocab_path: Path | str | None = None,
) -> Any:
    if vocab_path is not None:
        path = Path(vocab_path)
        if not path.exists():
            raise SystemExit(f"Provided vocab_path not found: {path}")
    else:
        path = resolve_tokenizer_path_from_metadata(checkpoint)
    tokenizer = load_tokenizer(path)
    _validate_tokenizer_metadata(
        tokenizer=tokenizer,
        tokenizer_path=path,
        checkpoint=checkpoint,
    )

    expected_num_embed = checkpoint_num_embed(checkpoint, variables=variables)
    if expected_num_embed is not None and tokenizer.vocab_size != expected_num_embed:
        raise SystemExit(
            "Tokenizer/checkpoint vocab mismatch: "
            f"tokenizer has {tokenizer.vocab_size} tokens, checkpoint expects "
            f"{expected_num_embed}. Check checkpoint metadata vocab_path."
        )

    args.tokenizer = path
    return tokenizer


def resolve_tokenizer_path_from_metadata(checkpoint: Any) -> Path:
    metadata = checkpoint_metadata(checkpoint)
    raw_vocab_path = metadata.get("vocab_path")
    if isinstance(raw_vocab_path, str) and raw_vocab_path.strip():
        for path in _metadata_path_candidates(raw_vocab_path):
            path = path.resolve()
            if path.exists():
                return path

        raise SystemExit(
            f"Checkpoint metadata vocab_path not found: {raw_vocab_path}. "
            "Save the checkpoint with a portable vocab_path in metadata."
        )

    raise SystemExit(
        "Checkpoint metadata is missing vocab_path. Cannot load tokenizer without vocab metadata."
    )


def _metadata_path_candidates(raw_path: str) -> list[Path]:
    raw_text = str(raw_path).strip()
    path = Path(raw_text).expanduser()
    candidates = [path if path.is_absolute() else REPO_ROOT / path]

    parts = [part for part in raw_text.replace("\\", "/").split("/") if part]
    for anchor in ("src", REPO_ROOT.name):
        if anchor not in parts:
            continue
        start = parts.index(anchor)
        suffix = parts[start:] if anchor == "src" else parts[start + 1 :]
        if suffix:
            candidates.append(REPO_ROOT.joinpath(*suffix))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _validate_tokenizer_metadata(
    *,
    tokenizer: Any,
    tokenizer_path: Path,
    checkpoint: dict[str, Any],
) -> None:
    metadata = checkpoint_metadata(checkpoint)

    raw_size = metadata.get("vocab_size")
    if raw_size is not None and int(raw_size) != int(tokenizer.vocab_size):
        raise SystemExit(
            f"Tokenizer vocab_size {tokenizer.vocab_size} != checkpoint metadata "
            f"vocab_size {raw_size}."
        )

    raw_sha256 = metadata.get("vocab_sha256")
    if isinstance(raw_sha256, str) and raw_sha256.strip():
        actual = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
        if actual != raw_sha256.strip():
            raise SystemExit(
                f"Tokenizer sha256 mismatch for {tokenizer_path}: "
                f"{actual} != checkpoint metadata {raw_sha256}."
            )


def _resolve_num_embed(
    *,
    config: Any,
    tokenizer: Any,
    checkpoint: dict[str, Any],
    variables: dict[str, Any],
) -> Any:
    expected_num_embed = checkpoint_num_embed(checkpoint, variables=variables)
    expected_num_embed = tokenizer.vocab_size if expected_num_embed is None else expected_num_embed

    if expected_num_embed != config.num_embed:
        raise SystemExit(
            f"Checkpoint num_embed mismatch: metadata config has {config.num_embed}, "
            f"params/tokenizer expect {expected_num_embed}."
        )

    if tokenizer.vocab_size != config.num_embed:
        raise SystemExit(
            f"Tokenizer size {tokenizer.vocab_size} != config.num_embed {config.num_embed}."
        )

    return config


def _resolve_logits_head(
    *,
    config: Any,
    checkpoint: dict[str, Any],
    variables: dict[str, Any],
) -> Any:
    logits_head = checkpoint_logits_head(checkpoint, variables=variables)
    if logits_head != config.logits_head:
        raise SystemExit(
            f"Checkpoint logits_head mismatch: metadata config has "
            f"{config.logits_head!r}, params imply {logits_head!r}."
        )
    return config
