# Linear Equation Problem Contract

Locks the current linear-equation domain behavior:

- Pretraining samples are conditioned by a unique equation id prompt.
- Pretraining targets use compact step-by-step algebra.
- SFT samples use the Vietnamese natural-language prompt with `?`.
- SFT targets explain chuyển vế đổi dấu and keep fractions unless reducible.
- SFT records are generated from a subset of the pretraining equations.
- Problem code is independent from the generic evaluator.
