#!/usr/bin/env python

"""Compute episode lengths for a LeRobot or Hugging Face dataset and write them to JSON."""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict


def try_import_lerobot_dataset() -> Any:
    """Return ``LeRobotDataset`` if importable, else ``None``."""
    try:
        from lerobot.dataset import LeRobotDataset # type: ignore

        return LeRobotDataset
    except Exception:
        return None


def try_import_datasets() -> Any:
    """Return the ``datasets`` module if importable, else ``None``."""
    try:
        import datasets # type: ignore

        return datasets
    except ImportError:
        return None


def load_dataset(dataset_id: str) -> Any:
    """Load a dataset via LeRobot first, falling back to the ``datasets`` library."""
    lr_ds = try_import_lerobot_dataset()
    if lr_ds is not None:
        try:
            return lr_ds.load(dataset_id)
        except Exception:
            pass

    ds_lib = try_import_datasets()
    if ds_lib is not None:
        try:
            return ds_lib.load_dataset(dataset_id, split="train")
        except Exception:
            pass

    return None


def compute_lengths(ds: Any) -> Dict[int, int]:
    """Iterate the dataset and return ``{episode_index: length}`` sorted by index."""
    lengths: Dict[int, int] = defaultdict(int)
    try:
        for sample in ds:
            epi = int(sample.get("episode_index", 0))
            frame_idx = int(sample.get("frame_index", sample.get("step_index", 0)))
            lengths[epi] = max(lengths[epi], frame_idx + 1)
    except Exception as exc:
        print(f"Error while iterating dataset: {exc}")
    return dict(sorted(lengths.items(), key=lambda kv: kv[0]))


def main() -> int:
    """Parse arguments, compute episode lengths, and write them to JSON."""
    parser = argparse.ArgumentParser(description="Compute episode lengths for a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset ID or path")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    ds = load_dataset(args.dataset)
    if ds is None:
        print("Failed to load dataset. Please ensure dependencies are installed.")
        return 1

    lengths = compute_lengths(ds)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(lengths, handle, indent=2)
        print(f"Wrote {len(lengths)} episode lengths to {args.out}")
        return 0
    except Exception as exc:
        print(f"Error writing output: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
