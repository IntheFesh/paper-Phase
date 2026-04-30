"""Round-1 diagnostic helper package.

Used only by ``scripts/diagnostics/diagnostic_phase_centric.py`` and
``scripts/smoke/smoke_test_diagnostic.py``. It is not imported by the
main repository code and therefore does not affect the existing training
or inference pipeline.

Modules:
    - phase_proxies: gripper / velocity_change / planner_output phase boundary proxies.
    - synthetic_env: SyntheticLongHorizonEnv (3-waypoint navigation) fallback for H2.
    - synthetic_demos: Synthetic demo generator (for offline pipeline smoke tests).
    - h1_loss: Per-timestep flow-matching loss extractor (no data aug).
    - h2_rollout: Episode rollout + misalignment computation.
    - report: JSON / Markdown / PNG output writers.
"""
