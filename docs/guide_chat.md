# Guide: Chat

Chat UI nam trong:

```text
apps/terminal/
```

## Install

```powershell
cd apps\terminal
bun install
```

Chay UI:

```powershell
cd apps\terminal
bun run chat
```

Checkpoint list mac dinh:

```text
recipes/chat/checkpoints.yaml
```

## Override Checkpoint

```powershell
cd apps\terminal
bun run chat -- `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

## Worker

Worker stdio dung cho UI hoac tool ben ngoai:

```powershell
cd apps\terminal
bun run chat:worker -- `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

## HTTP server

```powershell
python src/mintdim_lab/cli/main.py serve `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

## Boundary

- `apps/terminal`: terminal UI.
- `apps/electron`: Electron desktop UI boundary.
- `mintdim_lab.cli.commands.serve`: public HTTP server CLI.
- `mintdim_lab.serving.worker_cli`: JSON stdio worker.
- `mintdim_lab.inference.model_loader`: load model/checkpoint/tokenizer.
- `mintdim_lab.inference.text_generator`: generation loop and sampling.
