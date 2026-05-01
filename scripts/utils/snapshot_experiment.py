#!/usr/bin/env python
"""Package a small artifact tarball for an experiment directory.

After every PACE v2 experiment finishes, the launch script calls this
utility to copy a *small* subset of the experiment outputs to a
long-term storage location (typically ``/root/autodl-tmp/snapshots/``
on autodl). The tarball is intentionally narrow so it's cheap to ship
off-box: large model checkpoints stay on the training node, while the
analysis-relevant artifacts (CSV, Markdown report, eval JSON, figures)
travel.

Inclusion list (anything else is silently skipped):

- ``training_dynamics.csv``
- ``diagnostic_report.md``
- ``eval_results.json``
- everything inside ``figures/`` ending in ``.png`` or ``.pdf``

Usage
-----
::

    python scripts/utils/snapshot_experiment.py \\
        --src "$PACE_RUN_DIR/02_cliff_via_beta_only/seed_42" \\
        --dst "/root/autodl-tmp/snapshots/02_seed_42_$(date +%Y%m%d_%H%M%S).tar.gz"

The destination directory is created if missing. The script is
idempotent and never raises on a missing optional artifact — only the
files that exist are added.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from typing import Iterable, Optional, Sequence


# Top-level files (relative to ``--src``) included if they exist.
_INCLUDED_FILES = (
    "training_dynamics.csv",
    "diagnostic_report.md",
    "eval_results.json",
)

# Glob-style patterns inside ``figures/`` that are included.
_FIGURE_EXTENSIONS = (".png", ".pdf")

# Real checkpoint filenames (only included when --include_checkpoints is set).
_CHECKPOINT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "checkpoint_latest.pt",
)


def _figure_files(figures_dir: Path) -> Iterable[Path]:
    if not figures_dir.is_dir():
        return ()
    return (p for p in figures_dir.iterdir() if p.is_file() and p.suffix.lower() in _FIGURE_EXTENSIONS)


def build_snapshot(src: Path, dst: Path, include_checkpoints: bool = False) -> int:
    """Create a tar.gz at ``dst`` containing the inclusion list under ``src``.

    Returns the number of files actually added (0 means nothing matched
    and the tarball is therefore not created).

    When ``include_checkpoints`` is True, also pack the real model checkpoint
    files (``model.safetensors`` / ``pytorch_model.bin`` / ``checkpoint_latest.pt``).
    Useful for archiving final-stage checkpoints; off by default to keep the
    snapshot cheap to ship off-box.
    """
    if not src.is_dir():
        raise FileNotFoundError(f"snapshot source {src} is not a directory")

    members: list[tuple[Path, str]] = []
    for name in _INCLUDED_FILES:
        path = src / name
        if path.is_file():
            members.append((path, name))
    for fig in _figure_files(src / "figures"):
        members.append((fig, f"figures/{fig.name}"))
    if include_checkpoints:
        for name in _CHECKPOINT_FILES:
            path = src / name
            if path.is_file():
                members.append((path, name))

    if not members:
        print(f"[snapshot_experiment] no included artifacts under {src}; skipping {dst}")
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst, "w:gz") as tar:
        for path, arcname in members:
            tar.add(path, arcname=f"{src.name}/{arcname}")
    print(f"[snapshot_experiment] wrote {dst} ({len(members)} files)")
    return len(members)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Experiment directory to snapshot")
    p.add_argument("--dst", type=Path, required=True, help="Output .tar.gz path")
    p.add_argument(
        "--include_checkpoints",
        action="store_true",
        help="Also pack model.safetensors / pytorch_model.bin / checkpoint_latest.pt",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    n = build_snapshot(
        args.src.resolve(),
        args.dst.resolve(),
        include_checkpoints=args.include_checkpoints,
    )
    return 0 if n > 0 else 0  # never fails caller; absence of artifacts is not an error


if __name__ == "__main__":
    sys.exit(main())
