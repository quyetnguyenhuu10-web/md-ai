from __future__ import annotations

from typing import Any


def make_kv_cache(
    config: Any,
    *,
    batch_size: int = 1,
    max_seq_len: int | None = None,
    dtype: Any = None,
) -> tuple[dict[str, Any], ...]:
    import jax.numpy as jnp

    if dtype is None:
        dtype = jnp.float32

    cache_len = int(max_seq_len or config.max_seq_len)
    caches = []
    for layer_id in range(int(config.num_layers)):
        kv_heads = config.kv_heads_for_layer(layer_id)
        key_size = config.key_size_for_layer(layer_id)
        value_size = config.value_size_for_layer(layer_id)
        caches.append(
            {
                "key": jnp.zeros(
                    (
                        int(batch_size),
                        cache_len,
                        int(kv_heads),
                        int(key_size),
                    ),
                    dtype=dtype,
                ),
                "value": jnp.zeros(
                    (
                        int(batch_size),
                        cache_len,
                        int(kv_heads),
                        int(value_size),
                    ),
                    dtype=dtype,
                ),
            }
        )
    return tuple(caches)
