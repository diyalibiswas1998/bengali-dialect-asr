#!/usr/bin/env python
"""Package and push the maintained notebooks with the Kaggle CLI."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--notebook", choices=("creator", "training"), required=True)
    parser.add_argument("--processed-dataset", default=None, help="owner/slug; required for training")
    args = parser.parse_args()
    if not shutil.which("kaggle"):
        raise RuntimeError("Install the Kaggle CLI and configure ~/.kaggle/kaggle.json first")
    if args.notebook == "training" and not args.processed_dataset:
        raise ValueError("--processed-dataset owner/slug is required for the training notebook")

    repository = Path(__file__).resolve().parents[1]
    code_file = "kaggle_dataset_creator.ipynb" if args.notebook == "creator" else "kaggle_vaani_training.ipynb"
    slug = "vaani-bengali-processed-builder" if args.notebook == "creator" else "bengali-dialect-mms-moe-training"
    metadata = {
        "id": f"{args.username}/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": args.notebook == "training",
        "enable_internet": True,
        "dataset_sources": [args.processed_dataset] if args.processed_dataset else [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with tempfile.TemporaryDirectory(prefix="kaggle-kernel-") as temporary:
        package = Path(temporary)
        shutil.copy2(repository / code_file, package / code_file)
        (package / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        subprocess.check_call(["kaggle", "kernels", "push", "-p", str(package)])
    print(f"Pushed private Kaggle notebook: https://www.kaggle.com/code/{args.username}/{slug}")


if __name__ == "__main__":
    main()
