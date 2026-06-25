# addition_3term

Arithmetic JSONL dataset using:

```text
{prompt}\n{answer}<eos>
```

For two-term records, prompt is `x+y=?` and answer is `x+y=sum`.

For three-term records, prompt is `a+b+c?`.

Train split:

- `a+b`: 100 prompts, all digit pairs `0..9`.
- `ab+c`: 90 prompts, `ab` in `10..18`, `c` in `0..9`.
- `c+ab`: 90 reversed prompts.
- `a+b+c`: 220 prompts from the 1000 digit triples.

Eval split:

- remaining 780 `a+b+c` prompts.

For `a+b+c`, answer is step-by-step:

```text
a+b+c=(a+b)+c=sum
```
