# System Checkpoint Contract

Locks checkpoint system behavior that future code must preserve:

- Completed positive steps are checkpointed on the configured cadence.
- Step directories use deterministic zero-padded names.
- Retention pruning removes the oldest step directories only.
- Checkpoint paths remain simple filesystem paths independent of model code.
