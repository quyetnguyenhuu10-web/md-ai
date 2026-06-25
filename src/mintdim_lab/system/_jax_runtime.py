"""Shared implementation for strict JAX platform runtime adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PyTree = Any


def platform(platform_name: str) -> str:
    """Return the platform identity owned by a JAX runtime module."""
    return str(platform_name)


def devices(platform_name: str) -> tuple[Any, ...]:
    """Return all JAX-visible devices for one JAX platform."""
    import jax

    try:
        return tuple(jax.devices(backend=str(platform_name)))
    except RuntimeError:
        return ()


def local_devices(platform_name: str) -> tuple[Any, ...]:
    """Return devices for one JAX platform local to this process."""
    import jax

    try:
        return tuple(jax.local_devices(backend=str(platform_name)))
    except RuntimeError:
        return ()


def device_count(platform_name: str) -> int:
    """Return the number of JAX-visible devices for one JAX platform."""
    return len(devices(str(platform_name)))


def local_device_count(platform_name: str) -> int:
    """Return the number of process-local devices for one JAX platform."""
    return len(local_devices(str(platform_name)))


def require_device(platform_name: str, index: int = 0, *, local: bool = True) -> Any:
    """Return a device from one JAX platform or fail explicitly."""
    platform_name = str(platform_name)
    visible = local_devices(platform_name) if bool(local) else devices(platform_name)

    if not visible:
        scope = "local" if bool(local) else "visible"
        raise RuntimeError(f"No {platform_name.upper()} devices are {scope} to this process.")

    idx = int(index)
    if idx < 0 or idx >= len(visible):
        scope = "local" if bool(local) else "visible"
        raise IndexError(
            f"{platform_name.upper()} device index {idx} is out of range for "
            f"{len(visible)} {scope} device(s)."
        )

    return visible[idx]


def runtime_info(platform_name: str) -> dict[str, Any]:
    """Return raw JAX runtime information for one JAX platform."""
    import jax

    platform_name = str(platform_name)
    return {
        "platform": platform_name,
        "default_backend": jax.default_backend(),
        "device_count": device_count(platform_name),
        "local_device_count": local_device_count(platform_name),
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
    }


def put_tree(tree: PyTree, device: Any) -> PyTree:
    """Place an arbitrary PyTree on an explicitly supplied device."""
    import jax

    return jax.device_put(tree, device)


def compile_callable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Compile a generic callable with JAX JIT."""
    import jax

    return jax.jit(fn)


def sync(value: Any) -> Any:
    """Block until a generic JAX value is ready."""
    import jax

    return jax.block_until_ready(value)


def make_runtime_api(platform_name: str) -> dict[str, Callable[..., Any]]:
    """Return module-level runtime functions bound to one JAX platform."""
    platform_name = str(platform_name)

    def bound_platform() -> str:
        return platform(platform_name)

    def bound_devices() -> tuple[Any, ...]:
        return devices(platform_name)

    def bound_local_devices() -> tuple[Any, ...]:
        return local_devices(platform_name)

    def bound_device_count() -> int:
        return device_count(platform_name)

    def bound_local_device_count() -> int:
        return local_device_count(platform_name)

    def bound_require_device(index: int = 0, *, local: bool = True) -> Any:
        return require_device(platform_name, index=index, local=local)

    def bound_runtime_info() -> dict[str, Any]:
        return runtime_info(platform_name)

    return {
        "platform": bound_platform,
        "devices": bound_devices,
        "local_devices": bound_local_devices,
        "device_count": bound_device_count,
        "local_device_count": bound_local_device_count,
        "require_device": bound_require_device,
        "runtime_info": bound_runtime_info,
        "put_tree": put_tree,
        "compile_callable": compile_callable,
        "sync": sync,
    }


__all__ = [
    "PyTree",
    "compile_callable",
    "device_count",
    "devices",
    "local_device_count",
    "local_devices",
    "make_runtime_api",
    "platform",
    "put_tree",
    "require_device",
    "runtime_info",
    "sync",
]
