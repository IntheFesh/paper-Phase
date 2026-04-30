"""PhaseQFlow policy package.

This package exposes lightweight configuration/utilities without forcing heavy
runtime deps at import time. Model/processor classes are lazily imported.
"""

from .configuration_phaseqflow import PhaseQFlowConfig

__all__ = [
    "PhaseQFlowConfig",
    "PhaseQFlowPolicy",
    "PhaseQFlowProcessor",
]


def __getattr__(name: str):
    """Lazily import heavy runtime modules to keep package import lightweight."""
    if name == "PhaseQFlowPolicy":
        from .modeling_phaseqflow import PhaseQFlowPolicy

        return PhaseQFlowPolicy
    if name == "PhaseQFlowProcessor":
        from .processor_pc_vla import PhaseQFlowProcessor

        return PhaseQFlowProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
