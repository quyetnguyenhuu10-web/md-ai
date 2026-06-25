"""Strict TPU JAX runtime adapter."""

from __future__ import annotations

from mintdim_lab.system._jax_runtime import make_runtime_api

_API = make_runtime_api("tpu")

platform = _API["platform"]
devices = _API["devices"]
local_devices = _API["local_devices"]
device_count = _API["device_count"]
local_device_count = _API["local_device_count"]
require_device = _API["require_device"]
runtime_info = _API["runtime_info"]
put_tree = _API["put_tree"]
compile_callable = _API["compile_callable"]
sync = _API["sync"]

__all__ = [
    "platform",
    "devices",
    "local_devices",
    "device_count",
    "local_device_count",
    "require_device",
    "runtime_info",
    "put_tree",
    "compile_callable",
    "sync",
]
