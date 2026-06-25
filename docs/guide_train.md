# Guide: Train

Training entrypoint chinh la:

```powershell
python src/mintdim_lab/cli/main.py train --repo .
```

## Short Smoke

```powershell
python src/mintdim_lab/cli/main.py train --repo . --max-steps 5 --no-compile-update
```

## Config

```text
recipes/train/cpu/linear_equation_tiny_cpu.yaml
```

Production recipe set:

```text
recipes/train/
recipes/models/
recipes/corpus/
recipes/tokenizers/
```

Runtime settings live inside each `recipes/train/*.yaml` file.

## Outputs

Default checkpoint directory:

```text
runs/train/linear_equation_tiny_cpu/checkpoints/
```

Optional metrics:

```powershell
python src/mintdim_lab/cli/main.py train --repo . --jsonl runs\train_metrics.jsonl
```

Optional determinism debug log:

```powershell
python src/mintdim_lab/cli/main.py train --repo . --determinism-log runs\determinism.jsonl
```

## Boundary

- `mintdim_lab.cli`: parse CLI args only.
- `mintdim_lab.run_setup`: resolve configs and assemble run context.
- `mintdim_lab.trainer`: objective, accumulation, update loop, checkpoint policy.
- `mintdim_lab.system`: device/runtime/checkpoint IO.
