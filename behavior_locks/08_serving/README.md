# Serving Contract

Locks inference/serving behavior:

- Chat prompts default to raw text unless formatting is explicitly requested.
- Deterministic sampling (`temperature <= 0`) returns argmax and does not split RNG.
- Runtime loading remains text-only at the model boundary.
- Production serving code lives in `mintdim_lab.serving`; legacy serving
  and chat-worker import paths are retired.
