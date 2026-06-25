from __future__ import annotations

from pathlib import Path

from mintdim_lab.run_setup import bundle as run_setup_bundle
from mintdim_lab.run_setup import resolver

REPO = Path(__file__).resolve().parents[2]
TEST_MODEL_CONFIG = "recipes/models/test_tiny.yaml"


def test_run_setup_bundle_loads_training_objects_directly():
    resolved = run_setup_bundle.load_training_bundle(
        repo=REPO,
        model_config=TEST_MODEL_CONFIG,
    )

    assert resolved.training.model_config.endswith(".yaml")
    assert resolved.model_config.num_embed == resolved.tokenizer.vocab_size
    assert tuple(entry.unit for entry in resolved.unit_read.queue) == (96,)
    assert callable(resolved.update_step)


def test_run_setup_bundle_accepts_unit_read_corpus_directory_override():
    resolved = run_setup_bundle.load_training_bundle(
        repo=REPO,
        model_config=TEST_MODEL_CONFIG,
        unit_read_config="recipes/corpus/linear_equation_unit96",
    )

    assert tuple(entry.unit for entry in resolved.unit_read.queue) == (96,)


def test_run_setup_bundle_exposes_training_setup_entrypoints():
    assert run_setup_bundle.TrainingBundle.__module__ == "mintdim_lab.run_setup.bundle"
    assert callable(run_setup_bundle.load_training_bundle)
    assert callable(run_setup_bundle.learning_rate_schedule)
    assert callable(run_setup_bundle.build_optimizer)
    assert callable(run_setup_bundle.build_model_apply)
    assert callable(run_setup_bundle.init_train_state)


def test_run_setup_resolver_preserves_config_override_semantics(tmp_path):
    repo = tmp_path
    training_config = repo / "recipes" / "training.yaml"
    override = repo / "recipes" / "override.yaml"

    assert resolver.resolve_config_path(repo, training_config) == training_config
    assert resolver.resolve_config_path(repo, "recipes/training.yaml") == training_config
    assert (
        resolver.config_override_or_training_value(
            repo=repo,
            override=override,
            training_value="recipes/training.yaml",
        )
        == override
    )
    assert (
        resolver.config_override_or_training_value(
            repo=repo,
            override=None,
            training_value="recipes/training.yaml",
        )
        == training_config
    )
