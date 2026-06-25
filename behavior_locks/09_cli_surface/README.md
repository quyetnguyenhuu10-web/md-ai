# CLI Surface Contract

Locks the public command-line surface:

- Top-level CLI dispatches build-units, train, train-tokenizer, eval, params,
  chat, and serve.
- Train CLI exposes config overrides, runtime choice, JSON metrics, checkpoint
  directory, and optional determinism debug log.
- Params CLI reports model/tokenizer metadata from config without training.
- Production CLI code lives under `mintdim_lab.cli`; legacy CLI import
  paths are retired.
