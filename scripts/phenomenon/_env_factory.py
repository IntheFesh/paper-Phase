"""Shared env-factory helper for phenomenon scripts.

The phenomenon scripts (universality, regret_scaling, triangulation_concordance)
all need a way to roll out a policy in a real LIBERO environment. This module
provides one entry point — :func:`build_libero_env_factory` — which returns a
zero-arg callable that constructs a fresh LIBERO env for a given task name, or
``None`` if the env cannot be built (e.g. ``lerobot`` missing or task unknown).

The callers should treat ``None`` as a signal to fall back to dry-run synthetic
data and emit a warning, rather than crashing.
"""

from __future__ import annotations

from typing import Callable, Optional


def build_libero_env_factory(
    task: str = "libero_long",
    seed: int = 0,
) -> Optional[Callable]:
    """Return a zero-arg callable that constructs a LIBERO env, or ``None``.

    Parameters
    ----------
    task : str
        LIBERO task suite name (``libero_long``, ``libero_spatial``,
        ``libero_object``, ``libero_goal``).
    seed : int
        Seed forwarded to the env constructor.

    Returns
    -------
    callable or None
        A function with signature ``() -> env`` returning a fresh env every
        call.  ``None`` if the environment cannot be constructed (the caller
        should fall back to dry-run synthetic data).
    """
    try:
        from lerobot.envs.factory import make_env  # type: ignore
    except ImportError:
        try:
            from lerobot.common.envs.factory import make_env  # type: ignore
        except ImportError:
            return None

    def _factory():
        try:
            return make_env(env_type="libero", task=task, seed=seed)
        except Exception:  # noqa: BLE001
            return None

    probe = _factory()
    if probe is None:
        return None
    try:
        probe.close()
    except Exception:  # noqa: BLE001
        pass
    return _factory
