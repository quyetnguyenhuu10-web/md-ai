# Evaluator Contract

Locks benchmark/evaluation behavior:

- Prompt rendering stops before the configured target field.
- The target field is the expected answer for exact scoring.
- Exact scoring flags control line endings, case, and whitespace.
- Reports expose compact wrong-answer records.
- Production evaluation code lives in `mintdim_lab.evaluator`; legacy
  benchmark import paths are retired.
- Evaluator stays generic and must not import `mintdim_lab.problems` or know
  any concrete problem package such as `linear_equation`.
