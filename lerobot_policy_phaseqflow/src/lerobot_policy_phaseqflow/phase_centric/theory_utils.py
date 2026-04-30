"""Shared math helpers for the Phase-Centric subpackage.

This module holds the small mathematical primitives that multiple phase-centric
innovations need: run-length posteriors, Shannon entropy, Jensen-Shannon
divergence, empirical CDFs. Keeping them here avoids re-implementing the same
formulas in ``identifiability.py`` and ``pcar_trigger.py``.

Current state
-------------
Placeholders only. Each helper will be filled in the first time a caller
actually needs it (lazy migration).

Expected API sketch
-------------------
- ``run_length_posterior(prior, likelihood) -> torch.Tensor``
- ``shannon_entropy(probs, dim=-1) -> torch.Tensor``
- ``empirical_cdf(samples, grid) -> np.ndarray``
"""

from __future__ import annotations

from typing import Any


def run_length_posterior(*args: Any, **kwargs: Any) -> Any:
    """Placeholder; will be filled when first needed by PCAR."""

    raise NotImplementedError("run_length_posterior: lazy fill-in.")


def shannon_entropy(*args: Any, **kwargs: Any) -> Any:
    """Placeholder; will be filled when first needed by PACE-A entropy regulariser."""

    raise NotImplementedError("shannon_entropy: lazy fill-in.")


def empirical_cdf(*args: Any, **kwargs: Any) -> Any:
    """Placeholder reserved for future statistical diagnostics."""

    raise NotImplementedError("empirical_cdf: lazy fill-in.")
