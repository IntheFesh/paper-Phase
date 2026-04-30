#!/usr/bin/env python

"""Export a LeRobot checkpoint directory into a standalone artifact by copying core files."""

import argparse
import os
import shutil
from typing import List


CORE_FILES = ["config.json", "model.safetensors"]


def export_checkpoint(src: str, dst: str) -> int:
    """Copy the recognised core files from ``src`` into ``dst``; return a process exit code."""
    if not os.path.isdir(src):
        print(f"Source path does not exist: {src}")
        return 1

    os.makedirs(dst, exist_ok=True)

    copied: List[str] = []
    missing: List[str] = []

    for filename in CORE_FILES:
        src_file = os.path.join(src, filename)
        if not os.path.isfile(src_file):
            missing.append(src_file)
            continue

        dst_file = os.path.join(dst, filename)
        try:
            shutil.copy2(src_file, dst_file)
            copied.append(dst_file)
            print(f"Copied {src_file} -> {dst_file}")
        except Exception as exc:
            print(f"Error copying {src_file}: {exc}")
            return 1

    if missing:
        print("Missing expected files:")
        for path in missing:
            print(f"- {path}")

    if not copied:
        print("No files were copied.")
        return 1

    print(f"Export complete: copied {len(copied)} file(s) to {dst}")
    return 0


def main() -> int:
    """Parse CLI arguments and run :func:`export_checkpoint`."""
    parser = argparse.ArgumentParser(description="Export a LeRobot checkpoint directory")
    parser.add_argument("--src", type=str, required=True, help="Source checkpoint directory")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    args = parser.parse_args()
    return export_checkpoint(args.src, args.dst)


if __name__ == "__main__":
    raise SystemExit(main())
