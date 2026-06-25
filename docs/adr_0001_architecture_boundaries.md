# ADR 0001: Architecture Boundaries

## Status

Accepted during Domain Driven refactor.

## Decision

Production Python code lives under `src/mintdim_lab`.

Top-level boundaries:

- `problems`: optional task/domain tooling: data synthesis, conversion, rendering, and domain-specific grading. It is separate from generic evaluation; ordinary file-based evaluation must not depend on it.
- `corpus`: build/read/batch data units.
- `tokenizer`: tokenizer config, runtime adapter, train, audit.
- `model`: Transformer architecture.
- `trainer`: objective, accumulation, update loop, metrics, checkpoint policy.
- `evaluator`: checkpoint-level evaluation engine. It should stay generic: load examples, render prompts from declared fields, generate predictions, score predictions, and write reports.
- `serving`: inference, chat worker, HTTP API.
- `run_setup`: resolve configs and assemble one run.
- `system`: path/env/device/checkpoint IO integration.
- `cli`: thin command entrypoints.

Non-production or local artifacts:

- `recipes`: runnable configs.
- `data_store`: local data artifacts.
- `runs`: generated outputs.
- `apps`: frontend/UI apps.
- `studies`: notebooks and experiments outside production path.

## Evaluation boundary

For ordinary language-model benchmarks, evaluation should be file based:

    examples file -> declared input fields -> prompt template -> generation -> target field -> scoring -> report

The evaluator must not require benchmarks to have a module under
`mintdim_lab.problems`. If a benchmark is expressible as input fields plus target
fields, keep it in evaluator config.

Use `mintdim_lab.problems` only for separate task/domain pipelines: synthesis,
conversion, special rendering, parsing, or domain-specific grading. The generic
evaluator must not import `problems`, and `problems` must not import the generic
evaluator.

## Legacy Exit

The retired `md` namespace has been removed. New production imports must use
`mintdim_lab.*`.

## Consequences

- New code should enter through the boundary that owns the behavior.
- Tests are grouped by boundary under `tests/`.
- UI code is no longer inside the retired Python package.
