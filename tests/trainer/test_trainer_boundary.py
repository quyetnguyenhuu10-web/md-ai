from __future__ import annotations

from mintdim_lab.trainer import checkpointing, config, reproducibility, terminal_ui


def test_trainer_config_boundary_exposes_config_entrypoints():
    assert config.TrainingConfig.__module__ == "mintdim_lab.trainer.config.training"
    assert config.OptimizationConfig.__module__ == "mintdim_lab.trainer.config.training"
    assert config.UnitReadConfig.__module__ == "mintdim_lab.trainer.config.unit_read"
    assert callable(config.build_training_config)
    assert callable(config.build_unit_read_config)
    assert callable(config.load_training_config_yaml)
    assert callable(config.load_unit_read_config_yaml)


def test_trainer_checkpointing_boundary_exposes_checkpoint_entrypoints():
    assert callable(checkpointing.should_save_checkpoint)
    assert callable(checkpointing.checkpoint_step_dir)
    assert callable(checkpointing.save_train_state_checkpoint)
    assert callable(checkpointing.latest_checkpoint_step)
    assert callable(checkpointing.list_checkpoint_steps)


def test_trainer_reproducibility_boundary_exposes_debug_entrypoints():
    assert callable(reproducibility.tree_sha256)
    assert callable(reproducibility.runtime_fingerprint)
    assert callable(reproducibility.unit_read_source_fingerprints)


def test_trainer_terminal_ui_boundary_exposes_terminal_entrypoint():
    assert terminal_ui.TerminalTrainUI.__module__ == "mintdim_lab.trainer.terminal_ui"
