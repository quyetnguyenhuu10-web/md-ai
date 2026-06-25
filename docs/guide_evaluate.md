# Guide: Eval

Eval is the generic checkpoint benchmark path.

It is meant for ordinary language-model evaluation:

```
examples file
-> declared input fields
-> prompt template
-> model generation
-> target field
-> exact / normalized-exact scoring
-> JSON report
```

Eval must not require any benchmark to have a module under `mintdim_lab.problems`.

Use `problems` only when building a separate task/domain pipeline outside the
generic eval path. If the benchmark is already expressible as input fields plus
a target field, keep it in the eval config.

## Current command

Run from repo root:

```
python src/mintdim_lab/cli/main.py eval --config recipes\evaluation\math_exact.yaml
```

For real checkpoint evaluation, usually pass the checkpoint explicitly:

```
python src/mintdim_lab/cli/main.py eval ^
  --config recipes\evaluation\math_exact.yaml ^
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 ^
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

PowerShell version:

```
python src/mintdim_lab/cli/main.py eval `
  --config recipes\evaluation\math_exact.yaml `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

## Quick smoke eval

Use `--limit` to test the pipeline quickly:

```
python src/mintdim_lab/cli/main.py eval `
  --config recipes\evaluation\math_exact.yaml `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json `
  --limit 10
```

This is useful after training to check that:

* the checkpoint can be loaded,
* the tokenizer path is correct,
* the eval JSONL file exists,
* generation runs,
* scoring produces a report.

## Write report to a custom path

Use `--json-path`:

```
python src/mintdim_lab/cli/main.py eval `
  --config recipes\evaluation\math_exact.yaml `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json `
  --json-path runs\eval\linear_equation_step_00000600.json
```

If `--json-path` is not passed, evaluator writes the report near the checkpoint, under a benchmark/report directory.

## Current YAML format

Current file:

```
recipes/evaluation/math_exact.yaml
```

Current schema is intentionally unchanged for now.

Example:

```
checkpoint:
  path: runs/train/linear_equation_tiny_cpu/checkpoints/step_00000600

vocab:
  path: data_store/tokenizers/byte_bpe_512/tokenizer.json

data:
  eval_path: data_store/raw/linear_equation/eval.jsonl

template:
  sequence:
    - "{prompt}\n{target}<eos>"
  input_until:
    field: target
  target:
    field: target

runtime:
  batch_size: 64
  max_new_tokens: 96
  fixed_seq_len: null
  limit: 1000
  stop_at_target_length: true

scoring:
  ignore_whitespace: false
  normalize_line_endings: false
  case_insensitive: false

output:
  wrong_limit: 20
```

## Config reference

### `checkpoint.path`

Path to one concrete checkpoint.

Example:

```
checkpoint:
  path: runs/train/linear_equation_tiny_cpu/checkpoints/step_00000600
```

Do not point this to the parent checkpoint root if the evaluator expects one concrete checkpoint.

Good:

```
runs/train/linear_equation_tiny_cpu/checkpoints/step_00000600
```

Usually wrong:

```
runs/train/linear_equation_tiny_cpu/checkpoints
```

You can override this from CLI:

```
--checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600
```

CLI override is recommended when the YAML contains an old or placeholder checkpoint path.

### `vocab.path`

Path to tokenizer JSON.

Example:

```
vocab:
  path: data_store/tokenizers/byte_bpe_512/tokenizer.json
```

You can override this from CLI:

```
--vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

Use this when evaluating a checkpoint trained with a specific tokenizer.

### `data.eval_path`

Path to eval examples.

Example:

```
data:
  eval_path: data_store/raw/linear_equation/eval.jsonl
```

The current common format is JSONL.

Each line should be one JSON object.

Example record:

```
{"prompt": "2x + 3 = 7\nx =", "target": "2"}
```

For normal model benchmarks, this is enough:

* one or more input fields,
* one target field,
* optional metadata fields.

The evaluator must not need problem-specific code for this case.

### `template.sequence`

A list of string fragments used to describe the full text layout.

Current example:

```
template:
  sequence:
    - "{prompt}\n{target}<eos>"
```

This means the full logical sequence is:

```
prompt text
newline
target text
eos token
```

The template can reference fields from each JSONL record using braces.

Example:

```
"{question}\nAnswer: {answer}<eos>"
```

would require fields named `question` and `answer`.

### `template.input_until.field`

The field that marks where model input stops.

Current example:

```
input_until:
  field: target
```

With the current sequence:

```
"{prompt}\n{target}<eos>"
```

and:

```
input_until.field: target
```

the evaluator builds the model prompt from the text before the target field.

So the target is part of the full training/eval sequence definition, but it is not supposed to be leaked into the model input.

Conceptually:

```
rendered full sequence: "{prompt}\n{target}<eos>"
model input:            text before target
expected answer:        target
```

### `template.target.field`

The field used as the gold answer.

Current example:

```
target:
  field: target
```

This means each JSONL record must contain a string field named `target`.

Example:

```
{"prompt": "2x + 3 = 7\nx =", "target": "2"}
```

The evaluator compares model prediction against this target after optional normalization.

### `runtime.batch_size`

Number of examples processed per eval batch.

Example:

```
batch_size: 64
```

Increase it for faster eval if memory allows.

Decrease it if eval runs out of memory.

### `runtime.max_new_tokens`

Maximum number of tokens the model may generate for each example.

Example:

```
max_new_tokens: 96
```

For exact-answer tasks, this can usually be small.

For longer generation tasks, increase it.

### `runtime.fixed_seq_len`

Optional fixed sequence length.

Example:

```
fixed_seq_len: null
```

When null, evaluator can use dynamic or inferred sizing depending on implementation.

Keep this null unless a specific model/runtime path needs a fixed sequence length.

### `runtime.limit`

Maximum number of eval examples to load/run from the dataset.

Example:

```
limit: 1000
```

Use smaller values for smoke tests.

Example CLI override:

```
--limit 10
```

### `runtime.stop_at_target_length`

Whether generation should stop once it reaches the target length.

Example:

```
stop_at_target_length: true
```

For exact-match tasks, this is useful because the expected answer length is known.

For open-ended generation eval, this may be too restrictive.

### `scoring.ignore_whitespace`

Whether to ignore all whitespace differences during exact matching.

Example:

```
ignore_whitespace: false
```

If false, whitespace matters except trailing whitespace may still be stripped by the normalizer.

Use true when answers like these should count the same:

```
"x = 2"
"x=2"
```

or:

```
"hello world"
"helloworld"
```

Only enable this if whitespace is not meaningful for the task.

### `scoring.normalize_line_endings`

Whether to normalize line endings before comparison.

Example:

```
normalize_line_endings: false
```

Use true if answers may contain Windows or Unix line ending differences.

This helps treat these as equivalent:

```
"\r\n"
"\n"
```

### `scoring.case_insensitive`

Whether to lowercase prediction and target before comparison.

Example:

```
case_insensitive: false
```

Use true when capitalization should not matter.

Example:

```
"Paris"
"paris"
```

For code, math symbols, or case-sensitive tasks, keep this false.

### `output.wrong_limit`

Maximum number of wrong examples to include in the report sample.

Example:

```
wrong_limit: 20
```

This does not change scoring. It only controls how many wrong examples are shown in the output report for debugging.

## Common use cases

### Evaluate the latest trained checkpoint manually

Find the latest checkpoint directory under:

```
runs/train/linear_equation_tiny_cpu/checkpoints/
```

Then pass it explicitly:

```
python src/mintdim_lab/cli/main.py eval `
  --config recipes\evaluation\math_exact.yaml `
  --checkpoint runs\train\linear_equation_tiny_cpu\checkpoints\step_00000600 `
  --vocab-path data_store\tokenizers\byte_bpe_512\tokenizer.json
```

### Evaluate another JSONL dataset with same fields

If the new dataset still uses:

```
prompt
target
```

then only change:

```
data:
  eval_path: path/to/other_eval.jsonl
```

Everything else can stay the same.

### Evaluate JSONL with different field names

Suppose your JSONL is:

```
{"question": "2 + 2 =", "answer": "4"}
```

Then the template should reference those fields:

```
template:
  sequence:
    - "{question}\n{answer}<eos>"
  input_until:
    field: answer
  target:
    field: answer
```

The model input is the text before `answer`.

The expected output is the `answer` field.

### Exact match task

Use:

```
scoring:
  ignore_whitespace: false
  normalize_line_endings: false
  case_insensitive: false
```

This is strict and good for tasks where exact formatting matters.

### Normalized exact task

Use one or more normalization options:

```
scoring:
  ignore_whitespace: true
  normalize_line_endings: true
  case_insensitive: true
```

This is useful when surface formatting should not matter.

## Recommended future simplified config

The current YAML is supported, but the intended simpler model is:

```
checkpoint:
  path: runs/train/linear_equation_tiny_cpu/checkpoints/step_00000600

vocab:
  path: data_store/tokenizers/byte_bpe_512/tokenizer.json

data:
  format: jsonl
  path: data_store/raw/linear_equation/eval.jsonl
  input_fields:
    - prompt
  target_field: target

prompt:
  template: "{prompt}"

generation:
  batch_size: 64
  max_new_tokens: 96
  limit: 1000
  stop_at_target_length: true

scoring:
  type: norm_exact
  ignore_whitespace: false
  normalize_line_endings: false
  case_insensitive: false

output:
  wrong_limit: 20
```

This future format is easier to understand, but it is not required for the current phase.

## Boundary rule

Keep eval generic.

Good eval responsibilities:

* read examples,
* render prompt,
* run model generation,
* compare prediction to target,
* write summary/report.

Bad eval responsibilities:

* knowing what linear equations are,
* generating task data,
* parsing task-specific math,
* embedding custom domain grading into the generic runner.

Those belong in `mintdim_lab.problems` only when you are building a separate
task/domain pipeline outside generic eval.

For ordinary JSONL input/target benchmarks, do not use `problems`.
