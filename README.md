# mintdim-lab

Local JAX training lab for MintDim text language models. Run commands from the
repository root.

## Install

```powershell
python -m pip install -e .
cd apps\terminal
bun install
cd ..\..
```

`bun install` installs only the chat UI dependencies declared in
`apps/terminal/package.json`; `node_modules` belongs inside `apps/terminal`.
All commands below run directly from the repository root through `src/mintdim_lab/cli/main.py`.

## Layout

- `src/mintdim_lab/`: production Python package.
- `recipes/`: runnable YAML recipes for training, model, corpus, tokenizer, evaluation, and chat.
- `data_store/`: local raw data, packed unit shards, and tokenizer artifacts.
- `apps/terminal/`: Bun/OpenTUI terminal chat interface.
- `apps/electron/`: Electron desktop app boundary.
- `studies/notebooks/`: notebooks and non-production studies.
- `runs/`: local outputs from training, evaluation, and chat.

`mintdim_lab.problems` is for separate task/domain pipelines when generic
JSONL evaluation is not enough. Generic `eval` does not depend on `problems`.

## Build Training Units

Build tokenized `.bin` shards from `data_store/raw/linear_equation/pretrain_sft.jsonl`:

```powershell
python src/mintdim_lab/cli/main.py build-units recipes\corpus\linear_equation_unit96
```

The command reads `recipes/corpus/linear_equation_unit96/unit_build.yaml`, uses
the public `mintdim` unit-build API, and writes artifacts under
`data_store/packed/linear_equation_unit96`.

You can also pass an explicit unit-build YAML file:

```powershell
python src/mintdim_lab/cli/main.py build-units recipes\corpus\linear_equation_unit96\unit_build.yaml
```

## Train Tokenizer

Train a Byte-BPE tokenizer from JSONL text fields:

```powershell
python src/mintdim_lab/cli/main.py train-tokenizer `
  --input data_store\raw\linear_equation\pretrain_sft.jsonl `
  --output-dir data_store\tokenizers\byte_bpe_512
```

## Train

Train on CPU and write one metrics record after every completed optimizer
update:

```powershell
python src/mintdim_lab/cli/main.py train --repo . `
  --training-config recipes\train\cpu\linear_equation_tiny_cpu.yaml `
  --jsonl runs\train\linear_equation_tiny_cpu\train_metrics.jsonl
```

Use another train recipe for another JAX runtime:

```powershell
python src/mintdim_lab/cli/main.py train --repo . `
  --training-config recipes\train\gpu\linear_equation_tiny_gpu.yaml `
  --jsonl runs\train\linear_equation_tiny_gpu\train_metrics.jsonl

python src/mintdim_lab/cli/main.py train --repo . `
  --training-config recipes\train\tpu\linear_equation_tiny_tpu.yaml `
  --jsonl runs\train\linear_equation_tiny_tpu\train_metrics.jsonl
```

Run a short CPU smoke test:

```powershell
python src/mintdim_lab/cli/main.py train --repo . --max-steps 5 --no-compile-update
```

Training reads `recipes/train/cpu/linear_equation_tiny_cpu.yaml` by default.
That training recipe composes model, tokenizer, corpus, optimizer, and
runtime settings from the `recipes/` tree. Checkpoints are written under
`runs/train/linear_equation_tiny_cpu/checkpoints` by default.

Each JSONL metrics row contains:

- `step`: completed optimizer updates, counted from `1`.
- `step_index`: Python loop position, counted from `0`.
- `effective_batch_size`: examples in the accumulation window.
- `token_count`: target tokens learned during that update.
- `loss_mean`: mean training loss for that update.

## Evaluate

Evaluate the checkpoint configured in `recipes/evaluation/math_exact.yaml`:

```powershell
python src/mintdim_lab/cli/main.py eval --config recipes\evaluation\math_exact.yaml
```

Evaluate another checkpoint without editing YAML:

```powershell
python src/mintdim_lab/cli/main.py eval `
  --config recipes\evaluation\math_exact.yaml `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json `
  --json-path runs\eval\math-step600.json
```

Use `--limit` for a quick check:

```powershell
python src/mintdim_lab/cli/main.py eval --config recipes\evaluation\math_exact.yaml --limit 10
```

`--checkpoint` must point to one concrete `step_*` directory or checkpoint file,
not a parent checkpoint directory.

## Count Parameters

Count the exact parameter shapes selected by a model YAML and tokenizer YAML:

```powershell
python src/mintdim_lab/cli/main.py params `
  --model-config recipes\models\tiny.yaml `
  --tokenizer-config recipes\tokenizers\byte_bpe_512.yaml
```

The tokenizer config determines the model embedding table size, so the reported
total includes the real embedding table selected for that run.

Default tokenizer artifact: `data_store/tokenizers/byte_bpe_512/tokenizer.json`.

## Chat

Launch the Bun TUI. It reads the first usable checkpoint from
`recipes/chat/checkpoints.yaml` unless you pass an explicit checkpoint:

```powershell
cd apps\terminal
bun run chat
```

Use another checkpoint without editing YAML:

```powershell
cd apps\terminal
bun run chat -- `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000700 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

The UI launches Python internally. Override the Python command locally with
`--python <command>` or `MINTDIM_PYTHON` when your environment needs it.

Run the Python JSON stdio worker directly only when another UI/process will
send JSON commands to stdin:

```powershell
cd apps\terminal
bun run chat:worker -- `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000700 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

## Serve

Run the HTTP inference server:

```powershell
python src/mintdim_lab/cli/main.py serve `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

## CLI Help

```powershell
python src/mintdim_lab/cli/main.py
python src/mintdim_lab/cli/main.py build-units --help
python src/mintdim_lab/cli/main.py train --help
python src/mintdim_lab/cli/main.py train-tokenizer --help
python src/mintdim_lab/cli/main.py eval --help
python src/mintdim_lab/cli/main.py params --help
python src/mintdim_lab/cli/main.py chat --help
python src/mintdim_lab/cli/main.py serve --help
```
