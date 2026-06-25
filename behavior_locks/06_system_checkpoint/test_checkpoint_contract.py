from __future__ import annotations

from mintdim_lab.trainer.checkpointing import (
    checkpoint_step_dir,
    latest_checkpoint_step,
    list_checkpoint_steps,
    prune_old_checkpoints,
    should_save_checkpoint,
)


def test_checkpoint_save_cadence_and_step_directory_contract(tmp_path):
    assert should_save_checkpoint(step=100, save_every=100)
    assert not should_save_checkpoint(step=0, save_every=100)
    assert not should_save_checkpoint(step=99, save_every=100)

    assert checkpoint_step_dir(tmp_path, step=7).name == "step_00000007"
    assert checkpoint_step_dir(tmp_path, step=600).name == "step_00000600"


def test_checkpoint_retention_prunes_oldest_steps_contract(tmp_path):
    for step in (100, 200, 300, 400):
        checkpoint_step_dir(tmp_path, step=step).mkdir(parents=True)

    assert list_checkpoint_steps(tmp_path) == (100, 200, 300, 400)
    assert latest_checkpoint_step(tmp_path) == 400

    deleted = prune_old_checkpoints(checkpoint_dir=tmp_path, max_to_keep=2)

    assert deleted == (100, 200)
    assert list_checkpoint_steps(tmp_path) == (300, 400)
    assert latest_checkpoint_step(tmp_path) == 400
