# Corpus Pipeline Contract

Locks the current corpus boundary:

- `unit_read` config is tokenizer-free and points at already-packed token ids.
- Training targets the `target` segment plus the final `token_template` segment.
- `target.index` is 0-based and must match `layout.sequence_template`.
- Unit-read records map to `Batch`.
- Accumulation windows stack as `[accum, batch, unit]`.
- Corpus source keeps `batch`, `formatting`, `mintdim_unit_build`, and
  `mintdim_unit_read`; source-level `manifest` and `schema` modules are retired.
