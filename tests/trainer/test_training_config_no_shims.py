from __future__ import annotations

import importlib.util


def test_training_config_has_no_split_module_shims():
    assert importlib.util.find_spec("md") is None
