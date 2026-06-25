# Model Contract

Locks the current text-model behavior:

- `recipes/models/test_tiny.yaml` is the test-only model YAML used by tests that
  need to instantiate or load a model config.
- Local/global attention branch fields remain explicit.
- RoPE/NoPE split dimensions are validated.
- Local/global branches resolve their own QK dim, RoPE dim, NoPE dim,
  theta, scale, KV heads, and QK logits softcap.
- Split RoPE/NoPE tensor math rotates the leading RoPE slice and preserves the
  trailing NoPE slice.
- The Transformer forward path returns finite logits with shape `[B, T, vocab]`.
- Production model code lives in `mintdim_lab.model`; legacy model import
  paths are retired.
