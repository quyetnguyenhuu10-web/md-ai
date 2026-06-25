# Tokenizer Contract

Locks the current tokenizer boundary:

- Training tokenizer config is `hf_json`.
- Vocab metadata is derived from HuggingFace `tokenizer.json`.
- Required special ids are stable for the current repo tokenizer.
- Runtime tokenizer can encode/decode text and optionally append EOS.
- Production tokenizer code lives in `mintdim_lab.tokenizer`; legacy
  tokenizer import paths are retired.
