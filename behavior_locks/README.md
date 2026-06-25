# Behavior Locks

This directory contains executable behavior contracts for the current repo.

They are intentionally separate from the normal `tests/` suite so a future
Domain Driven rewrite can use them as a focused reference pack:

```powershell
pytest behavior_locks -q
```

Each subdirectory maps to one proposed production boundary and contains:

- `test_*.py`: pytest tests that lock observable behavior.
- `README.md`: a short explanation of the behavior being preserved.

These tests should describe behavior, not the final folder layout.
