"""Generate the Kaggle T4x2 training/resume/evaluation notebook."""

import json
from pathlib import Path


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "cells": [
        markdown("""# MMS-300M Bengali dialect MoE — resumable T4×2 training

Attach the processed private Vaani Dataset. For later sessions, also attach the previous run checkpoint Dataset and set `PRIOR_RUN_DIR`. Keep Internet enabled so the first session can retrieve MMS-300M.
"""),
        code("""import os, shutil, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/diyalibiswas1998/bengali-dialect-asr.git"
REPO_DIR = Path("/kaggle/working/bengali-dialect-asr")
if not REPO_DIR.exists():
    subprocess.check_call(["git", "clone", REPO_URL, str(REPO_DIR)])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)])
"""),
        code("""# Auto-detect the attached processed Vaani Dataset.
dataset_candidates = [
    path for path in Path("/kaggle/input").iterdir()
    if (path / "metadata.json").exists() and (path / "vocab.json").exists()
]
if len(dataset_candidates) != 1:
    raise RuntimeError(
        "Attach exactly one processed Vaani Dataset; found "
        f"{[str(path) for path in dataset_candidates]}"
    )
DATASET_DIR = dataset_candidates[0]
PRIOR_RUN_DIR = None  # Example: Path("/kaggle/input/bengali-moe-checkpoints/moe-run")
RUN_DIR = Path("/kaggle/working/moe-run")
EXPERIMENT = "moe"  # baseline, moe, top1, no_dialect, or no_shared

if PRIOR_RUN_DIR and not RUN_DIR.exists():
    shutil.copytree(PRIOR_RUN_DIR, RUN_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"experiment={EXPERIMENT} data={DATASET_DIR} output={RUN_DIR}")
"""),
        code("""# Mandatory two-process forward/backward and exact state-restoration check.
subprocess.check_call([
    "accelerate", "launch", "--config_file", str(REPO_DIR / "configs/accelerate_t4x2.yaml"),
    str(REPO_DIR / "scripts/smoke_test_research.py"),
    "--config", str(REPO_DIR / "configs/research.yaml"),
    "--data-dir", str(DATASET_DIR), "--require-two-gpus",
])
"""),
        code("""command = [
    "accelerate", "launch", "--config_file", str(REPO_DIR / "configs/accelerate_t4x2.yaml"),
    str(REPO_DIR / "scripts/train_research.py"),
    "--config", str(REPO_DIR / "configs/research.yaml"),
    "--data-dir", str(DATASET_DIR),
    "--output-dir", str(RUN_DIR),
    "--experiment", EXPERIMENT,
]
if list(RUN_DIR.glob("checkpoint-*")):
    command += ["--resume", "latest"]
subprocess.check_call(command)
"""),
        code("""# Evaluate only when all three phases are present.
final_checkpoint = RUN_DIR / "checkpoint-phase-3"
if final_checkpoint.exists():
    subprocess.check_call([
        sys.executable, str(REPO_DIR / "scripts/validate_checkpoint.py"),
        "--checkpoint", str(final_checkpoint), "--expected-processes", "2",
    ])
    subprocess.check_call([
        "accelerate", "launch", "--config_file", str(REPO_DIR / "configs/accelerate_t4x2.yaml"),
        str(REPO_DIR / "scripts/evaluate_research.py"),
        "--checkpoint", str(final_checkpoint), "--data-dir", str(DATASET_DIR),
    ])
else:
    print("This session ended before phase 3. Save RUN_DIR as a private Kaggle Dataset version and resume next session.")
"""),
        markdown("""After every Kaggle session, save `RUN_DIR` as a new private Dataset version. Keep separate run directories for the baseline, main MoE, and each ablation. The split fingerprints in every `config.json` must match before comparing results.
"""),
    ],
}

code_root = Path(__file__).resolve().parents[1]
for destination in (code_root / "kaggle_vaani_training.ipynb", code_root.parent / "kaggle_vaani_training.ipynb"):
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("Generated Kaggle training notebooks.")
