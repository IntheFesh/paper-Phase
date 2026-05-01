#!/usr/bin/env python
"""Write a ``manifest.json`` describing a PACE v2 H800 training run.

Captures git revision, package versions, GPU info, the skip/dry-run
filter lists, and the launch timestamp. The cloud and local launch
scripts call this exactly once at the start of every run; downstream
analysis (and the final aggregated report) reads the manifest to
explain *why* a particular row is missing or marked placeholder.

Usage
-----
::

    python scripts/utils/write_manifest.py --output $PACE_RUN_DIR/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _run(cmd: Sequence[str]) -> Optional[str]:
    """Run a shell command and return its stripped stdout, or ``None`` if
    the command is missing or fails."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out.strip() or None


def _git_info() -> Dict[str, Any]:
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": (_run(["git", "status", "--porcelain"]) or "") != "",
        "remote": _run(["git", "config", "--get", "remote.origin.url"]),
    }


def _python_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for pkg in ("torch", "numpy", "transformers", "lerobot"):
        info[pkg] = _package_version(pkg)
    return info


def _package_version(pkg: str) -> Optional[str]:
    try:
        from importlib import metadata

        return metadata.version(pkg)
    except Exception:  # pragma: no cover  (e.g. missing pkg)
        return None


def _gpu_info() -> List[Dict[str, Any]]:
    """Best-effort GPU inventory via ``nvidia-smi``."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus: List[Dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]) if parts[2].isdigit() else parts[2],
                "driver_version": parts[3],
                "compute_capability": parts[4],
            }
        )
    return gpus


def _split_csv_env(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_manifest() -> Dict[str, Any]:
    """Assemble the manifest dict; pure function for testability."""
    return {
        "schema": "pace_v2/manifest/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": os.environ.get("PACE_RUN_DIR"),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "git": _git_info(),
        "env": _python_env(),
        "gpus": _gpu_info(),
        "skip_lists": {
            "ablations": _split_csv_env("PACE_SKIP_ABLATIONS"),
            "phenomena": _split_csv_env("PACE_SKIP_PHENOMENA"),
        },
        "dry_run_lists": {
            "ablations": _split_csv_env("PACE_DRYRUN_ABLATIONS"),
            "phenomena": _split_csv_env("PACE_DRYRUN_PHENOMENA"),
        },
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Path to write manifest.json")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[write_manifest] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
