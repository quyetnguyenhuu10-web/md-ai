from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mintdim_lab.model.config import TransformerConfig, make_attention_types

REQUIRED_TEXT_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "attention",
        "normalization",
        "output",
        "runtime",
    }
)

OPTIONAL_TEXT_FIELDS: frozenset[str] = frozenset()

LEGACY_IGNORED_TEXT_FIELDS: frozenset[str] = frozenset(
    {
        "mtp_num_future_tokens",
        "mtp_loss_weight",
    }
)


def check_required_fields(
    values: Mapping[str, Any],
    *,
    required: frozenset[str],
    label: str,
) -> None:
    """Validate that a config mapping contains all required fields."""
    missing = sorted(name for name in required if name not in values)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{label} is missing required text config fields: {missing_text}")


def _legacy_zero(value: Any) -> bool:
    if value is None:
        return True
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def normalize_text_config_values(
    values: Mapping[str, Any],
    *,
    preset_name: str,
) -> dict[str, Any]:
    """Return a clean text config mapping accepted by build_text_config.

    Unknown keys are rejected so YAML typos do not silently change model shape.

    The old MTP fields are tolerated only when set to zero. This lets older YAML
    snippets still load after MTP was removed, while preventing nonzero MTP
    settings from being silently ignored.
    """
    data = dict(values)
    allowed = REQUIRED_TEXT_FIELDS | OPTIONAL_TEXT_FIELDS

    unknown = sorted(
        name for name in data if name not in allowed and name not in LEGACY_IGNORED_TEXT_FIELDS
    )
    if unknown:
        unknown_text = ", ".join(unknown)
        raise ValueError(f"{preset_name} contains unknown text config fields: {unknown_text}")

    for name in LEGACY_IGNORED_TEXT_FIELDS:
        if name not in data:
            continue
        value = data.pop(name)
        if not _legacy_zero(value):
            raise ValueError(
                f"{preset_name} contains removed MTP field {name}={value!r}; "
                "MTP has been removed, so this field must be omitted or set to zero."
            )

    return data


def _require_mapping(value: Any, *, field_name: str, preset_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{preset_name}.{field_name} must be a mapping")
    return value


def _check_mapping_keys(
    value: Mapping[str, Any],
    *,
    field_name: str,
    required: set[str],
    optional: set[str] | None = None,
    preset_name: str,
) -> None:
    optional = optional or set()
    unknown = sorted(name for name in value if name not in required and name not in optional)
    if unknown:
        unknown_text = ", ".join(f"{field_name}.{name}" for name in unknown)
        raise ValueError(f"{preset_name} contains unknown text config fields: {unknown_text}")

    missing = sorted(name for name in required if name not in value)
    if missing:
        missing_text = ", ".join(f"{field_name}.{name}" for name in missing)
        raise ValueError(f"{preset_name} is missing required text config fields: {missing_text}")


def _flatten_model_config(
    data: dict[str, Any],
    *,
    preset_name: str,
) -> None:
    model = _require_mapping(data["model"], field_name="model", preset_name=preset_name)
    _check_mapping_keys(
        model,
        field_name="model",
        required={
            "max_sequence_length",
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "activation",
            "tie_word_embeddings",
        },
        preset_name=preset_name,
    )

    data["max_seq_len"] = model["max_sequence_length"]
    data["num_layers"] = model["num_layers"]
    data["embed_dim"] = model["hidden_size"]
    data["dense_hidden_dim"] = model["intermediate_size"]
    data["ffw_activation"] = model["activation"]
    data["logits_head"] = "tied" if bool(model["tie_word_embeddings"]) else "untied"


def _flatten_attention_branch_config(
    attention: Mapping[str, Any],
    *,
    branch_name: str,
    yaml_name: str,
    has_window_size: bool,
    preset_name: str,
) -> dict[str, Any]:
    branch = _require_mapping(
        attention[yaml_name],
        field_name=f"attention.{yaml_name}",
        preset_name=preset_name,
    )
    required = {"num_key_value_heads", "qk_head_dim", "qk_logits_softcap", "rope"}
    if has_window_size:
        required.add("window_size")
    _check_mapping_keys(
        branch,
        field_name=f"attention.{yaml_name}",
        required=required,
        preset_name=preset_name,
    )

    rope = _require_mapping(
        branch["rope"],
        field_name=f"attention.{yaml_name}.rope",
        preset_name=preset_name,
    )
    _check_mapping_keys(
        rope,
        field_name=f"attention.{yaml_name}.rope",
        required={"dim", "theta", "scale"},
        preset_name=preset_name,
    )

    return {
        f"num_{branch_name}_kv_heads": branch["num_key_value_heads"],
        f"{branch_name}_qk_dim": branch["qk_head_dim"],
        f"{branch_name}_qk_logits_softcap": branch["qk_logits_softcap"],
        f"{branch_name}_rope_dim": rope["dim"],
        f"{branch_name}_rope_base": rope["theta"],
        f"{branch_name}_rope_scale": rope["scale"],
    }


def flatten_attention_config(
    values: Mapping[str, Any],
    *,
    preset_name: str,
) -> dict[str, Any]:
    data = dict(values)
    attention = _require_mapping(
        data["attention"],
        field_name="attention",
        preset_name=preset_name,
    )
    _check_mapping_keys(
        attention,
        field_name="attention",
        required={
            "num_attention_heads",
            "value_head_dim",
            "layer_types",
            "local_sliding",
            "global",
        },
        preset_name=preset_name,
    )

    data["num_heads"] = attention["num_attention_heads"]
    data["v_head_dim"] = attention["value_head_dim"]
    data["attention_pattern"] = attention["layer_types"]

    local = _flatten_attention_branch_config(
        attention,
        branch_name="local",
        yaml_name="local_sliding",
        has_window_size=True,
        preset_name=preset_name,
    )
    data.update(local)
    data["sliding_window_size"] = attention["local_sliding"]["window_size"]
    data.update(
        _flatten_attention_branch_config(
            attention,
            branch_name="global",
            yaml_name="global",
            has_window_size=False,
            preset_name=preset_name,
        )
    )
    data.pop("attention")
    return data


def flatten_normalization_config(
    values: Mapping[str, Any],
    *,
    preset_name: str,
) -> dict[str, Any]:
    data = dict(values)
    normalization = _require_mapping(
        data["normalization"],
        field_name="normalization",
        preset_name=preset_name,
    )
    _check_mapping_keys(
        normalization,
        field_name="normalization",
        required={"post_attention", "post_ffn", "qk_norm"},
        preset_name=preset_name,
    )
    qk_norm = _require_mapping(
        normalization["qk_norm"],
        field_name="normalization.qk_norm",
        preset_name=preset_name,
    )
    _check_mapping_keys(
        qk_norm,
        field_name="normalization.qk_norm",
        required={"enabled", "learnable_scale"},
        preset_name=preset_name,
    )

    data["use_post_attn_norm"] = normalization["post_attention"]
    data["use_post_ffw_norm"] = normalization["post_ffn"]
    data["qk_norm_enabled"] = qk_norm["enabled"]
    data["qk_norm_with_scale"] = qk_norm["learnable_scale"]
    data.pop("normalization")
    return data


def flatten_output_config(
    values: Mapping[str, Any],
    *,
    preset_name: str,
) -> dict[str, Any]:
    data = dict(values)
    output = _require_mapping(data["output"], field_name="output", preset_name=preset_name)
    _check_mapping_keys(
        output,
        field_name="output",
        required={"vocab_logits_softcap"},
        preset_name=preset_name,
    )
    data["final_logit_softcap"] = output["vocab_logits_softcap"]
    data.pop("output")
    return data


def flatten_runtime_config(
    values: Mapping[str, Any],
    *,
    preset_name: str,
) -> dict[str, Any]:
    data = dict(values)
    runtime = _require_mapping(data["runtime"], field_name="runtime", preset_name=preset_name)
    _check_mapping_keys(
        runtime,
        field_name="runtime",
        required={"gradient_checkpointing", "flash_attention", "fused_attention"},
        preset_name=preset_name,
    )
    data["use_gradient_checkpointing"] = runtime["gradient_checkpointing"]
    data["use_flash_attention"] = runtime["flash_attention"]
    data["use_fused_attention"] = runtime["fused_attention"]
    data.pop("runtime")
    return data


def flatten_model_config(
    values: Mapping[str, Any],
    *,
    preset_name: str,
) -> dict[str, Any]:
    data = dict(values)
    _flatten_model_config(data, preset_name=preset_name)
    data.pop("model")
    data = flatten_attention_config(data, preset_name=preset_name)
    data = flatten_normalization_config(data, preset_name=preset_name)
    data = flatten_output_config(data, preset_name=preset_name)
    data = flatten_runtime_config(data, preset_name=preset_name)
    return data


def build_text_config(
    values: Mapping[str, Any],
    *,
    preset_name: str,
    num_embed: int,
    max_seq_len: int | None = None,
    num_layers: int | None = None,
) -> TransformerConfig:
    data = normalize_text_config_values(values, preset_name=preset_name)

    check_required_fields(
        {key: value for key, value in data.items() if key not in OPTIONAL_TEXT_FIELDS},
        required=REQUIRED_TEXT_FIELDS,
        label=preset_name,
    )

    data = flatten_model_config(data, preset_name=preset_name)

    if max_seq_len is not None:
        data["max_seq_len"] = int(max_seq_len)
    if num_layers is not None:
        data["num_layers"] = int(num_layers)

    cfg = TransformerConfig(
        num_embed=int(num_embed),
        max_seq_len=int(data["max_seq_len"]),
        num_layers=int(data["num_layers"]),
        embed_dim=int(data["embed_dim"]),
        dense_hidden_dim=int(data["dense_hidden_dim"]),
        ffw_activation=str(data["ffw_activation"]).strip().lower(),
        num_heads=int(data["num_heads"]),
        num_local_kv_heads=int(data["num_local_kv_heads"]),
        num_global_kv_heads=(
            None if data["num_global_kv_heads"] is None else int(data["num_global_kv_heads"])
        ),
        v_head_dim=int(data["v_head_dim"]),
        attention_types=make_attention_types(
            data["attention_pattern"],
            num_layers=int(data["num_layers"]),
        ),
        sliding_window_size=int(data["sliding_window_size"]),
        final_logit_softcap=(
            None if data["final_logit_softcap"] is None else float(data["final_logit_softcap"])
        ),
        local_qk_logits_softcap=(
            None
            if data["local_qk_logits_softcap"] is None
            else float(data["local_qk_logits_softcap"])
        ),
        global_qk_logits_softcap=(
            None
            if data["global_qk_logits_softcap"] is None
            else float(data["global_qk_logits_softcap"])
        ),
        use_post_attn_norm=bool(data["use_post_attn_norm"]),
        use_post_ffw_norm=bool(data["use_post_ffw_norm"]),
        qk_norm_enabled=bool(data["qk_norm_enabled"]),
        qk_norm_with_scale=bool(data["qk_norm_with_scale"]),
        local_qk_dim=int(data["local_qk_dim"]),
        global_qk_dim=int(data["global_qk_dim"]),
        local_rope_dim=int(data["local_rope_dim"]),
        global_rope_dim=int(data["global_rope_dim"]),
        local_rope_base=int(data["local_rope_base"]),
        global_rope_base=int(data["global_rope_base"]),
        local_rope_scale=float(data["local_rope_scale"]),
        global_rope_scale=float(data["global_rope_scale"]),
        logits_head=str(data["logits_head"]).strip().lower(),
        use_gradient_checkpointing=bool(data["use_gradient_checkpointing"]),
        use_fused_attention=bool(data.get("use_fused_attention", False)),
        use_flash_attention=bool(data.get("use_flash_attention", False)),
    )
    cfg.validate()
    return cfg


def read_text_config_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML model config file as a mapping."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML model configs.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        raise ValueError(f"{config_path} is empty.")
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level.")

    return dict(loaded)


def load_text_config_yaml(
    path: str | Path,
    *,
    num_embed: int,
    preset_name: str | None = None,
    max_seq_len: int | None = None,
    num_layers: int | None = None,
) -> TransformerConfig:
    """Load a TransformerConfig from a YAML model config file.

    num_embed is supplied by the caller because it normally comes from the
    tokenizer/vocabulary, not from the model-shape YAML.
    """
    config_path = Path(path)
    values = read_text_config_yaml(config_path)
    label = preset_name if preset_name is not None else config_path.stem

    return build_text_config(
        values,
        preset_name=label,
        num_embed=num_embed,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
    )


def load_text_config_file(
    path: str | Path,
    *,
    num_embed: int,
    preset_name: str | None = None,
    max_seq_len: int | None = None,
    num_layers: int | None = None,
) -> TransformerConfig:
    """Load a text config file.

    Currently supports .yaml and .yml files.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported text config file extension {suffix!r}; expected .yaml or .yml"
        )

    return load_text_config_yaml(
        path,
        num_embed=num_embed,
        preset_name=preset_name,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
    )


def _text_metadata_fields(*, include_num_embed: bool = False) -> tuple[str, ...]:
    fields: list[str] = []
    for field in dataclasses.fields(TransformerConfig):
        if field.name == "num_embed" and not include_num_embed:
            continue
        if field.name == "attention_types":
            fields.append("attention_pattern")
        else:
            fields.append(field.name)
    return tuple(fields)


def text_metadata_fields(*, include_num_embed: bool = False) -> tuple[str, ...]:
    """Return text config metadata field names without registry side effects."""
    return _text_metadata_fields(include_num_embed=include_num_embed)


__all__ = [
    "LEGACY_IGNORED_TEXT_FIELDS",
    "OPTIONAL_TEXT_FIELDS",
    "REQUIRED_TEXT_FIELDS",
    "build_text_config",
    "load_text_config_file",
    "load_text_config_yaml",
    "normalize_text_config_values",
    "read_text_config_yaml",
    "text_metadata_fields",
]
