from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from mintdim_lab.system import checkpoint_io, checkpoint_store
from mintdim_lab.trainer import checkpointing


def test_checkpoint_store_owns_step_paths_and_trainer_boundary_matches(tmp_path):
    assert checkpoint_store.should_save_checkpoint(step=4, save_every=2)
    assert not checkpoint_store.should_save_checkpoint(step=0, save_every=2)
    assert checkpointing.should_save_checkpoint(step=4, save_every=2)

    expected = tmp_path / "step_00000007"
    assert checkpoint_store.checkpoint_step_dir(tmp_path, step=7) == expected
    assert checkpointing.checkpoint_step_dir(tmp_path, step=7) == expected


def test_checkpoint_store_retention_matches_trainer_checkpointing(tmp_path):
    for step in (1, 2, 3):
        checkpoint_store.checkpoint_step_dir(tmp_path, step=step).mkdir(parents=True)

    assert checkpoint_store.list_checkpoint_steps(tmp_path) == (1, 2, 3)
    assert checkpointing.latest_checkpoint_step(tmp_path) == 3

    assert checkpoint_store.prune_old_checkpoints(checkpoint_dir=tmp_path, max_to_keep=2) == (1,)
    assert checkpointing.list_checkpoint_steps(tmp_path) == (2, 3)


def test_checkpoint_store_private_portable_helper_matches_trainer_boundary():
    tree = {"params": {"w": jnp.asarray([1.0, 2.0])}}

    portable = checkpoint_store.portable_array_tree(tree)
    trainer_portable = checkpointing.portable_array_tree(tree)

    assert isinstance(portable["params"]["w"], np.ndarray)
    assert isinstance(trainer_portable["params"]["w"], np.ndarray)


def test_checkpoint_io_owns_loading_helpers():
    ckpt = {"params": {"w": 1}, "metadata": {"step": 7}}

    assert checkpoint_io.checkpoint_to_variables(ckpt) == {"params": {"w": 1}}
    assert checkpoint_io.checkpoint_metadata(ckpt) == {"step": 7}

    resolved = checkpoint_io.resolve_checkpoint_path(Path("."))
    assert resolved == checkpoint_io.resolve_checkpoint_path(Path("."))
    assert ".npz" in checkpoint_io.CHECKPOINT_SUFFIXES
