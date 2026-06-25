from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence


class AttentionType(enum.Enum):
    GLOBAL = "global"
    LOCAL_SLIDING = "local_sliding"


# =============================================================================
# Config object
# =============================================================================


def _attention_type(value: AttentionType | str) -> AttentionType:
    if isinstance(value, AttentionType):
        return value
    return AttentionType(str(value))


def make_attention_types(
    pattern: Sequence[AttentionType | str],
    *,
    num_layers: int,
) -> tuple[AttentionType, ...]:
    if not pattern:
        raise ValueError("attention pattern must not be empty")

    attention_pattern = tuple(_attention_type(value) for value in pattern)
    return tuple(attention_pattern[i % len(attention_pattern)] for i in range(int(num_layers)))


@dataclasses.dataclass(frozen=True)
class TransformerConfig:
    """Static text Transformer architecture config.

    max_seq_len is a context ceiling: it is the largest sequence length the
    model will accept. It is not a padding target.

    When training from unit_read .bin records, the actual input sequence length
    is the selected unit_read queue entry unit. unit_build has already padded
    each record to that unit length, so training should feed that unit length
    directly and must not pad batches up to max_seq_len.

    Attention backend flags are explicit:
    - use_flash_attention enables MintDim's portable custom blockwise
      FlashAttention path, implemented with JAX/XLA primitives and usable on
      any runtime that can compile those primitives.
    - use_fused_attention enables JAX dot_product_attention with
      implementation="cudnn", which is NVIDIA GPU/cuDNN flash attention only
      and is not a TPU backend.

    Q/K position handling uses a fixed split per attention layer type:
    - qk_dim is the projected Q/K head dimension.
    - rope_dim is the leading even-dimensional slice that receives RoPE.
    - qk_dim - rope_dim is the NoPE content slice.
    rope_dim=0 is pure NoPE; rope_dim=qk_dim is pure RoPE.
    """

    num_embed: int
    max_seq_len: int
    num_layers: int
    embed_dim: int
    dense_hidden_dim: int
    ffw_activation: str
    num_heads: int
    num_local_kv_heads: int
    num_global_kv_heads: int | None
    v_head_dim: int
    attention_types: tuple[AttentionType, ...]
    sliding_window_size: int
    final_logit_softcap: float | None
    local_qk_logits_softcap: float | None
    global_qk_logits_softcap: float | None
    use_post_attn_norm: bool
    use_post_ffw_norm: bool
    qk_norm_enabled: bool
    qk_norm_with_scale: bool
    local_qk_dim: int
    global_qk_dim: int
    local_rope_dim: int
    global_rope_dim: int
    local_rope_base: int
    global_rope_base: int
    local_rope_scale: float
    global_rope_scale: float
    logits_head: str
    use_gradient_checkpointing: bool
    use_fused_attention: bool = False
    use_flash_attention: bool = False

    def kv_heads_for_layer(self, layer_id: int) -> int:
        if (
            self.attention_types[int(layer_id)] == AttentionType.GLOBAL
            and self.num_global_kv_heads is not None
        ):
            return int(self.num_global_kv_heads)
        return int(self.num_local_kv_heads)

    def key_size_for_layer(self, layer_id: int) -> int:
        return self.qk_dim_for_layer(layer_id)

    def qk_dim_for_layer(self, layer_id: int) -> int:
        if self.attention_types[int(layer_id)] == AttentionType.GLOBAL:
            return int(self.global_qk_dim)
        return int(self.local_qk_dim)

    def rope_dim_for_layer(self, layer_id: int) -> int:
        if self.attention_types[int(layer_id)] == AttentionType.GLOBAL:
            return int(self.global_rope_dim)
        return int(self.local_rope_dim)

    def nope_dim_for_layer(self, layer_id: int) -> int:
        return self.qk_dim_for_layer(layer_id) - self.rope_dim_for_layer(layer_id)

    def rope_base_for_layer(self, layer_id: int) -> int:
        if self.attention_types[int(layer_id)] == AttentionType.GLOBAL:
            return int(self.global_rope_base)
        return int(self.local_rope_base)

    def rope_scale_for_layer(self, layer_id: int) -> float:
        if self.attention_types[int(layer_id)] == AttentionType.GLOBAL:
            return float(self.global_rope_scale)
        return float(self.local_rope_scale)

    def qk_logits_softcap_for_layer(self, layer_id: int) -> float | None:
        if self.attention_types[int(layer_id)] == AttentionType.GLOBAL:
            return self.global_qk_logits_softcap
        return self.local_qk_logits_softcap

    def value_size_for_layer(self, layer_id: int) -> int:
        return int(self.v_head_dim)

    def validate(self) -> None:
        if self.num_embed <= 0:
            raise ValueError("num_embed must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if len(self.attention_types) != self.num_layers:
            raise ValueError("attention_types length must equal num_layers")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.dense_hidden_dim <= 0:
            raise ValueError("dense_hidden_dim must be positive")
        if self.ffw_activation not in {"gelu", "geglu", "silu", "swiglu"}:
            raise ValueError("ffw_activation must be one of: gelu, geglu, silu, swiglu")
        if self.num_heads <= 0 or self.num_local_kv_heads <= 0:
            raise ValueError("num_heads and num_local_kv_heads must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.v_head_dim <= 0:
            raise ValueError("v_head_dim must be positive")
        if self.num_heads * self.v_head_dim != self.embed_dim:
            raise ValueError("num_heads * v_head_dim must equal embed_dim")
        if self.num_heads % self.num_local_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_local_kv_heads")

        if self.num_global_kv_heads is not None:
            if self.num_global_kv_heads <= 0:
                raise ValueError("num_global_kv_heads must be positive when set")
            if self.num_heads % self.num_global_kv_heads != 0:
                raise ValueError("num_heads must be divisible by num_global_kv_heads")

        for name, qk_dim, rope_dim in (
            ("local", self.local_qk_dim, self.local_rope_dim),
            ("global", self.global_qk_dim, self.global_rope_dim),
        ):
            qk_dim = int(qk_dim)
            rope_dim = int(rope_dim)
            if qk_dim <= 0:
                raise ValueError(f"{name}_qk_dim must be positive")
            if rope_dim < 0:
                raise ValueError(f"{name}_rope_dim must be >= 0")
            if rope_dim > qk_dim:
                raise ValueError(f"{name}_rope_dim must be <= {name}_qk_dim")
            if rope_dim > 0 and rope_dim % 2 != 0:
                raise ValueError(f"{name}_rope_dim must be even when positive")

        if self.sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive")
        if self.final_logit_softcap is not None and self.final_logit_softcap <= 0.0:
            raise ValueError("final_logit_softcap must be positive when set")
        for name, softcap in (
            ("local_qk_logits_softcap", self.local_qk_logits_softcap),
            ("global_qk_logits_softcap", self.global_qk_logits_softcap),
        ):
            if softcap is not None and softcap <= 0.0:
                raise ValueError(f"{name} must be positive when set")
        for name, base in (
            ("local_rope_base", self.local_rope_base),
            ("global_rope_base", self.global_rope_base),
        ):
            if int(base) <= 0:
                raise ValueError(f"{name} must be positive")

        for name, scale in (
            ("local_rope_scale", self.local_rope_scale),
            ("global_rope_scale", self.global_rope_scale),
        ):
            if float(scale) < 1.0:
                raise ValueError(f"{name} must be >= 1.0")

        if self.logits_head not in {"tied", "untied"}:
            raise ValueError("logits_head must be 'tied' or 'untied'")
