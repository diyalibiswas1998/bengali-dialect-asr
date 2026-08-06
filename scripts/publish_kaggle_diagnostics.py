#!/usr/bin/env python
"""Publish the maintained CTC diagnostic/tiny-overfit notebook privately."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_upload" / "ctc_tiny_overfit"


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    if importlib.util.find_spec("kaggle"):
        return [sys.executable, "-m", "kaggle"]
    raise RuntimeError("Install and authenticate the Kaggle CLI before publishing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--audio-dataset", required=True, help="owner/slug")
    parser.add_argument("--checkpoint-dataset", required=True, help="owner/slug")
    parser.add_argument(
        "--slug", default="bengali-ctc-confirmed-tiny-overfit",
        help="kernel slug below the selected username",
    )
    parser.add_argument(
        "--title", default="Bengali CTC Confirmed Tiny Overfit",
    )
    args = parser.parse_args()

    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Maintained notebook directory is missing: {SOURCE}")
    metadata_path = SOURCE / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "id": f"{args.username}/{args.slug}",
            "title": args.title,
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [args.audio_dataset, args.checkpoint_dataset],
            "machine_shape": "NvidiaTeslaT4",
        }
    )

    with tempfile.TemporaryDirectory(prefix="bengali-ctc-kaggle-") as temporary:
        staging = Path(temporary)
        for source_file in SOURCE.iterdir():
            if source_file.is_file():
                shutil.copy2(source_file, staging / source_file.name)
        (staging / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(kaggle_command() + ["kernels", "push", "-p", str(staging)], check=True)
        print(f"Published: https://www.kaggle.com/code/{args.username}/{args.slug}")


if __name__ == "__main__":
    main()
