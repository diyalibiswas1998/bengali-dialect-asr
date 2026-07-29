"""Generate the no-derived-dataset Kaggle T4x2 training notebook."""

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
        markdown("""# Direct original-Vaani MMS-300M dialect MoE training

This notebook streams the original 11 West Bengal Vaani configurations during every pass. It does **not** write a derived audio dataset. Audio is decoded in memory, mixed to mono when necessary, and resampled to 16 kHz only because MMS requires 16 kHz input.

Trade-offs: the stable speaker-hash split is globally disjoint but not globally stratified; a mid-pass resume replays and skips the earlier stream; Hugging Face availability directly affects training; and three full passes can exceed several Kaggle sessions.
"""),
        code("""import os, shutil, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/diyalibiswas1998/bengali-dialect-asr.git"
REPO_DIR = Path("/kaggle/working/bengali-dialect-asr")
if not REPO_DIR.exists():
    subprocess.check_call(["git", "clone", REPO_URL, str(REPO_DIR)])
else:
    subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)])

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
print("Repository ready; HF_TOKEN loaded without displaying it.")
"""),
        code("""# Configure this run. Restore a previous checkpoint Dataset between Kaggle sessions.
PRIOR_RUN_DIR = None  # Example: Path("/kaggle/input/direct-vaani-checkpoints/direct-moe-run")
RUN_DIR = Path("/kaggle/working/direct-moe-run")
EXPERIMENT = "moe"  # baseline, moe, top1, no_dialect, no_shared
RUN_SMOKE = True

if PRIOR_RUN_DIR and not RUN_DIR.exists():
    shutil.copytree(PRIOR_RUN_DIR, RUN_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"experiment={EXPERIMENT} output={RUN_DIR}")
"""),
        code("""# Verify original-stream access plus one forward/backward on both T4 GPUs.
if RUN_SMOKE:
    subprocess.check_call([
        "accelerate", "launch", "--config_file", str(REPO_DIR / "configs/accelerate_t4x2.yaml"),
        str(REPO_DIR / "scripts/smoke_direct_streaming.py"),
        "--config", str(REPO_DIR / "configs/direct_streaming.yaml"),
        "--require-two-gpus",
    ])
"""),
        code("""# Three direct dataset passes: frozen encoder, top-4 unfrozen, reduced encoder LR.
command = [
    "accelerate", "launch", "--config_file", str(REPO_DIR / "configs/accelerate_t4x2.yaml"),
    str(REPO_DIR / "scripts/train_direct_streaming.py"),
    "--config", str(REPO_DIR / "configs/direct_streaming.yaml"),
    "--output-dir", str(RUN_DIR),
    "--experiment", EXPERIMENT,
    "--require-two-gpus",
]
if list(RUN_DIR.glob("checkpoint-*")):
    command += ["--resume", "latest"]
subprocess.check_call(command)
"""),
        code("""final_checkpoint = RUN_DIR / "checkpoint-phase-3"
if final_checkpoint.exists():
    print(f"Training complete: {final_checkpoint}")
else:
    print("Session ended before phase 3. Save RUN_DIR as a private Kaggle Dataset and resume it next session.")
"""),
        markdown("""## Session handoff

The notebook checkpoints every 250 optimizer steps and at each phase boundary. After each Kaggle session, publish `/kaggle/working/direct-moe-run` as a private Dataset version. Attach it next session and set `PRIOR_RUN_DIR`. Resume is deterministic but must replay the original stream up to `batch_in_phase`, so phase-boundary resumes are much faster than mid-phase resumes.
"""),
    ],
}

root = Path(__file__).resolve().parents[1]
for destination in (root / "kaggle_direct_vaani_training.ipynb", root.parent / "kaggle_direct_vaani_training.ipynb"):
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("Generated direct-stream Kaggle training notebooks.")
