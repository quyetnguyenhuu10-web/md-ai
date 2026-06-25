# Guide: Repository Structure

This repository is a local JAX training lab for text language models.

## Stable layout

    mintdim__lab/
    |-- apps/
    |   |-- terminal/
    |   `-- electron/
    |-- data_store/
    |   |-- raw/
    |   |-- packed/
    |   `-- tokenizers/
    |-- recipes/
    |   |-- train/
    |   |-- evaluation/
    |   |-- corpus/
    |   |-- models/
    |   |-- tokenizers/
    |   `-- chat/
    |-- src/
    |   `-- mintdim_lab/
    |       |-- cli/
    |       |-- corpus/
    |       |-- evaluator/
    |       |-- run_setup/
    |       |-- model/
    |       |-- problems/
    |       |-- serving/
    |       |-- system/
    |       |-- tokenizer/
    |       `-- trainer/
    |-- tests/
    |-- docs/
    |-- runs/
    |-- studies/
    |-- pyproject.toml
    `-- README.md

## Naming policy

`recipes/train/` contains training run recipes.

`recipes/evaluation/` remains the evaluation recipe directory for now. The
current YAML schema is intentionally unchanged.

`src/mintdim_lab/evaluator/` remains the evaluation engine package. The CLI
command can still be named `eval`; the package name does not need to be renamed.

## Role of `problems`

`problems` is optional task/domain tooling. It is separate from the generic
evaluator.

Use it for:

- data synthesis,
- data conversion,
- special rendering,
- domain-specific grading.

Do not make ordinary JSONL evaluation depend on `problems`.

Use `problems` when you are building a separate task pipeline and do not want
to use the existing generic eval path. It should not be imported by
`mintdim_lab.evaluator`, and it should not import `mintdim_lab.evaluator`.

## Role of `evaluator`

`evaluator` is a generic checkpoint benchmark engine.

Its basic job is:

    load checkpoint
    load tokenizer
    load examples
    render prompt from declared fields
    generate prediction
    score prediction against target
    write report

For normal model benchmarks, the evaluator should only need an examples file,
field declarations, generation settings, and a scorer such as exact or
normalized exact.

If a task needs domain-specific parsing, synthesis, or grading, keep that in
`problems` or in a separate runner instead of putting it into the generic
evaluator.
