#!/usr/bin/env python
"""Generate the full-test, imbalance-aware Kaggle evaluation notebook."""

import json
from pathlib import Path


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "cells": [
        markdown("""# Full evaluation — Bengali four-dialect MMS-300M MoE

This notebook evaluates the **full test split** from `diyalibiswas/four-dialect-of-bengali-covering-11-district`. It discovers all saved phase checkpoints from the training notebook output and selects the checkpoint with the lowest saved phase-end validation loss; test labels are never used for checkpoint selection.

Outputs include overall/macro/weighted/worst-group CER and WER, per-dialect and per-district ASR, imbalance-aware dialect classification (balanced accuracy, macro/weighted F1, MCC, Cohen kappa, G-mean, ROC-AUC, PR-AUC, calibration and normalized confusion), permutation-invariant router clustering/utilization, boundary-district sensitivity, stratified bootstrap confidence intervals, predictions, CSV tables, and plots.

Attach both the audio Dataset and the successful training notebook output. Select **GPU T4 x2** and enable Internet. This performs evaluation directly—there is no smoke-test subset.
"""),
        code("""import importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/diyalibiswas1998/bengali-dialect-asr.git"
REPO_DIR = Path("/kaggle/working/bengali-dialect-asr")
RUN_DIR = Path("/kaggle/working/four-dialect-evaluation")
METRICS_DIR = RUN_DIR / "metrics"
LOG_DIR = RUN_DIR / "logs"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SETUP_EXIT_CODE = 0

def run_logged(command, filename):
    log_path = LOG_DIR / filename
    print(f"Running command; persistent log: {log_path}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=environment,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()

try:
    git_command = (
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)]
        if not REPO_DIR.exists()
        else ["git", "-C", str(REPO_DIR), "pull", "--ff-only"]
    )
    if run_logged(git_command, "setup.log") != 0:
        raise RuntimeError("Git setup failed")
    requirements = {
        "accelerate": "accelerate>=0.31,<2",
        "datasets": "datasets>=2.20,<4",
        "matplotlib": "matplotlib>=3.7",
        "omegaconf": "omegaconf>=2.1",
        "safetensors": "safetensors>=0.4",
        "sklearn": "scikit-learn>=1.3",
        "soundfile": "soundfile>=0.12",
        "transformers": "transformers>=4.41,<5",
    }
    missing = [package for module, package in requirements.items() if importlib.util.find_spec(module) is None]
    if missing and run_logged([sys.executable, "-m", "pip", "install", "-q", *missing], "setup.log") != 0:
        raise RuntimeError("Runtime dependency installation failed")
    if run_logged(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)],
        "setup.log",
    ) != 0:
        raise RuntimeError("Editable project installation failed")
    os.environ.pop("HF_TOKEN", None)
    os.environ["VAANI_ALLOW_HF_FALLBACK"] = "0"
    print("Evaluation code is ready; attached-input-only mode is enforced.")
except Exception as exc:
    SETUP_EXIT_CODE = 1
    message = f"Setup failed: {type(exc).__name__}: {exc}\\n"
    print(message, end="")
    with (LOG_DIR / "setup.log").open("a", encoding="utf-8") as log:
        log.write(message)
"""),
        code("""# Evaluation configuration.
import math
import torch

DATASET_SOURCE = "diyalibiswas/four-dialect-of-bengali-covering-11-district"
TRAINING_NOTEBOOK_SOURCE = "diyalibiswas/bengali-four-dialect-mms-moe-500-steps"
SPLIT = "test"
BOOTSTRAP_ITERATIONS = 1000
PROGRESS_EVERY = 200
CONFIG_EXIT_CODE = 0
EVALUATION_EXIT_CODE = None

print("Evaluation target: full test split; no sample cap")
print("Checkpoint policy: lowest saved phase-end validation loss")
"""),
        code("""# Discover the attached dataset and select the best saved phase checkpoint.
SPLITS = ("train", "validation", "test")
input_root = Path("/kaggle/input")

def storage_mode(path):
    if not path.is_dir():
        return None
    if all((path / split).is_dir() for split in SPLITS):
        return "directories"
    if all((path / f"{split}.zip").is_file() for split in SPLITS):
        return "split-zips"
    return None

dataset_roots = set()
if input_root.is_dir():
    for item in input_root.rglob("test.zip"):
        if item.is_file() and storage_mode(item.parent):
            dataset_roots.add(item.parent)
    for item in input_root.rglob("test"):
        if item.is_dir() and storage_mode(item.parent):
            dataset_roots.add(item.parent)
dataset_candidates = sorted(dataset_roots, key=lambda path: str(path))
dataset_slug = DATASET_SOURCE.rsplit("/", 1)[-1].lower()
preferred_datasets = [path for path in dataset_candidates if dataset_slug in str(path).lower()]
if len(preferred_datasets) == 1:
    data_root = preferred_datasets[0]
elif len(dataset_candidates) == 1:
    data_root = dataset_candidates[0]
else:
    data_root = None
    CONFIG_EXIT_CODE = 1
    print(f"Could not uniquely locate audio data. Candidates={list(map(str, dataset_candidates))}")

checkpoint_candidates = []
if input_root.is_dir():
    for checkpoint in input_root.rglob("checkpoint-phase-*"):
        required = (
            checkpoint / "trainer_state.json",
            checkpoint / "config.json",
            checkpoint / "vocab.json",
        )
        model_exists = (checkpoint / "model_state.pt").is_file() or (checkpoint / "model.safetensors").is_file()
        if checkpoint.is_dir() and all(path.is_file() for path in required) and model_exists:
            try:
                state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
                validation_loss = float(state["validation_loss"])
                if math.isfinite(validation_loss):
                    checkpoint_candidates.append(
                        {
                            "path": checkpoint,
                            "validation_loss": validation_loss,
                            "global_step": int(state.get("global_step", 0)),
                            "complete": bool(state.get("complete", False)),
                        }
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

if checkpoint_candidates:
    checkpoint_candidates.sort(
        key=lambda item: (item["validation_loss"], -item["global_step"], str(item["path"]))
    )
    best_checkpoint = checkpoint_candidates[0]["path"]
else:
    best_checkpoint = None
    CONFIG_EXIT_CODE = 1
    print(
        "No valid phase checkpoint was found. Attach the output of "
        f"{TRAINING_NOTEBOOK_SOURCE}."
    )

gpu_info = [
    {
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "capability": list(torch.cuda.get_device_capability(index)),
    }
    for index in range(torch.cuda.device_count())
]
if len(gpu_info) != 2 or any(tuple(item["capability"]) < (7, 0) for item in gpu_info):
    CONFIG_EXIT_CODE = 1
    print(f"Wrong accelerator: {gpu_info}")
    print("Stop the session, choose GPU T4 x2 in Session options, restart, and Run All.")

selection = {
    "dataset_source": DATASET_SOURCE,
    "training_notebook_source": TRAINING_NOTEBOOK_SOURCE,
    "dataset_root": str(data_root) if data_root else None,
    "storage_mode": storage_mode(data_root) if data_root else None,
    "checkpoint_candidates": [
        {**item, "path": str(item["path"])} for item in checkpoint_candidates
    ],
    "selected_checkpoint": str(best_checkpoint) if best_checkpoint else None,
    "selection_rule": "minimum finite phase-end validation_loss; ties prefer later global_step",
    "gpu_info": gpu_info,
    "split": SPLIT,
    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
}
(RUN_DIR / "evaluation_selection.json").write_text(
    json.dumps(selection, indent=2), encoding="utf-8"
)
if CONFIG_EXIT_CODE == 0:
    os.environ["VAANI_AUDIO_ROOT"] = str(data_root)
    os.environ["VAANI_ALLOW_HF_FALLBACK"] = "0"
    os.environ.pop("VAANI_PARQUET_CACHE", None)
    os.environ.pop("VAANI_LOCAL_CONFIG", None)
    print(f"Dataset: {data_root} ({storage_mode(data_root)})")
    print("Phase checkpoints by validation loss:")
    for candidate in checkpoint_candidates:
        print(
            f"  {candidate['path'].name}: validation_loss={candidate['validation_loss']:.6f} "
            f"global_step={candidate['global_step']}"
        )
    print(f"Selected checkpoint: {best_checkpoint}")
    print(f"GPUs: {gpu_info}")
"""),
        code("""# Evaluate the full test split on both T4 GPUs.
if SETUP_EXIT_CODE == 0 and CONFIG_EXIT_CODE == 0:
    command = [
        "accelerate", "launch",
        "--config_file", str(REPO_DIR / "configs/accelerate_t4x2.yaml"),
        str(REPO_DIR / "scripts/evaluate_direct.py"),
        "--checkpoint", str(best_checkpoint),
        "--output-dir", str(METRICS_DIR),
        "--split", SPLIT,
        "--bootstrap-iterations", str(BOOTSTRAP_ITERATIONS),
        "--progress-every", str(PROGRESS_EVERY),
        "--require-two-gpus",
    ]
    EVALUATION_EXIT_CODE = run_logged(command, "evaluation.log")
else:
    print("Evaluation did not start because setup, input discovery, or GPU validation failed.")
print(f"evaluation_exit_code={EVALUATION_EXIT_CODE}")
"""),
        code("""# Persist metrics, predictions, plots, configuration, and logs as Kaggle outputs.
import datetime

summary_path = METRICS_DIR / "evaluation_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
manifest = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "dataset_source": DATASET_SOURCE,
    "training_notebook_source": TRAINING_NOTEBOOK_SOURCE,
    "split": SPLIT,
    "full_test_evaluation": True,
    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    "selected_checkpoint": str(best_checkpoint) if best_checkpoint else None,
    "setup_exit_code": SETUP_EXIT_CODE,
    "config_exit_code": CONFIG_EXIT_CODE,
    "evaluation_exit_code": EVALUATION_EXIT_CODE,
    "complete": EVALUATION_EXIT_CODE == 0 and summary is not None,
    "summary": summary,
    "metric_files": sorted(path.name for path in METRICS_DIR.glob("*")),
    "logs": sorted(path.name for path in LOG_DIR.glob("*.log")),
}
(RUN_DIR / "evaluation_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
archive = shutil.make_archive(
    "/kaggle/working/four-dialect-evaluation-artifacts", "zip", root_dir=RUN_DIR
)
print(f"Evaluation artifacts: {RUN_DIR}")
print(f"Downloadable archive: {archive}")
if summary:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
elif CONFIG_EXIT_CODE != 0:
    print("Evaluation is not complete. Correct the attached inputs/GPU shown above and Run All.")
else:
    raise RuntimeError(f"Evaluation failed; inspect {LOG_DIR / 'evaluation.log'}")
"""),
        markdown("""## Output interpretation

The authoritative result is `four-dialect-evaluation/metrics/evaluation_test.json`; the smaller `evaluation_summary.json` contains headline metrics. Use macro/balanced metrics for conclusions because the test set is imbalanced. `predictions_test.csv` enables error analysis. Router expert IDs are latent, so router NMI/ARI/utilization are meaningful while direct router-to-dialect accuracy is not.
"""),
    ],
}

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"evaluation-{index:02d}"

root = Path(__file__).resolve().parents[1]
(root / "kaggle_four_dialect_evaluation.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
)
print("Generated kaggle_four_dialect_evaluation.ipynb")
