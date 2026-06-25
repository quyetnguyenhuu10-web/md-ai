from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

from mintdim_lab.trainer.checkpointing import (
    checkpoint_step_dir,
    latest_checkpoint_step,
    list_checkpoint_steps,
    portable_array_tree,
    prune_old_checkpoints,
    save_train_state_checkpoint,
    should_save_checkpoint,
)
from mintdim_lab.trainer.state import TrainState


def _state(step: int) -> TrainState:
    return TrainState(
        params={"w": jnp.asarray([1.0, 2.0])},
        opt_state={"count": jnp.asarray(step)},
        step=step,
    )


def test_should_save_checkpoint_uses_completed_positive_steps():
    assert should_save_checkpoint(step=0, save_every=2) is False
    assert should_save_checkpoint(step=1, save_every=2) is False
    assert should_save_checkpoint(step=2, save_every=2) is True
    assert should_save_checkpoint(step=4, save_every=2) is True


def test_orbax_checkpoint_save_and_retention(tmp_path):
    root = tmp_path / "checkpoints"

    first = save_train_state_checkpoint(
        checkpoint_dir=root,
        state=_state(1),
        step=1,
        max_to_keep=2,
    )
    second = save_train_state_checkpoint(
        checkpoint_dir=root,
        state=_state(2),
        step=2,
        max_to_keep=2,
    )
    third = save_train_state_checkpoint(
        checkpoint_dir=root,
        state=_state(3),
        step=3,
        max_to_keep=2,
    )

    assert first == checkpoint_step_dir(root, step=1)
    assert second == checkpoint_step_dir(root, step=2)
    assert third == checkpoint_step_dir(root, step=3)

    assert not first.exists()
    assert second.is_dir()
    assert third.is_dir()
    assert list_checkpoint_steps(root) == (2, 3)
    assert latest_checkpoint_step(root) == 3

    metadata = json.loads((third / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["array_storage"] == "host_numpy"
    assert metadata["runtime_topology_dependent"] is False


def test_portable_array_tree_converts_device_arrays_to_host_numpy():
    tree = {
        "params": {"w": jnp.asarray([1.0, 2.0])},
        "state": (jnp.asarray(3, dtype=jnp.int32), "kept"),
    }

    portable = portable_array_tree(tree)

    assert isinstance(portable["params"]["w"], np.ndarray)
    assert isinstance(portable["state"][0], np.ndarray)
    assert portable["state"][1] == "kept"
    assert portable["params"]["w"].tolist() == [1.0, 2.0]
    assert portable["state"][0].item() == 3


def test_prune_old_checkpoints_rejects_nonpositive_retention(tmp_path):
    try:
        prune_old_checkpoints(checkpoint_dir=tmp_path, max_to_keep=0)
    except ValueError as exc:
        assert "max_to_keep must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
