# Trainer Contract

Locks the current training behavior:

- `basic.yaml` resolves model, tokenizer, unit-read, optimizer, and run cadence.
- Learning-rate schedule is 1-based for optimizer updates.
- Streaming accumulation performs one optimizer update after the full window.
- Effective batch size and token count are accumulated over micro-batches.
