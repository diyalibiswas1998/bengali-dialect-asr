#!/usr/bin/env python
"""Generate the local-input-only four-dialect Kaggle training notebook."""

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
        markdown("""# Local four-dialect Bengali MMS-300M MoE training

This notebook trains only on the attached private Kaggle Dataset `diyalibiswas/vaani-bengali-four-dialect-audio`, using its `train`, `validation`, and `test` WAV/TXT folders. It selects exactly 11 West Bengal districts and assigns them to Kamrupi, Jharkhandi, Varendri, and Rarhi. Dataset network fallback is disabled and no Hugging Face token is read. Internet is used only to clone the code and download the public MMS-300M model if it is not already cached.

The three training phases each stop at exactly 1,000 optimizer steps. Progress is printed and saved every 200 phase steps; resumable checkpoints are written every 100 optimizer steps and at phase boundaries. This notebook intentionally performs no preliminary test run.
"""),
        code("""import importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/diyalibiswas1998/bengali-dialect-asr.git"
REPO_DIR = Path("/kaggle/working/bengali-dialect-asr")
RUN_DIR = Path("/kaggle/working/local-four-dialect-run")
LOG_DIR = RUN_DIR / "logs"
RUN_DIR.mkdir(parents=True, exist_ok=True)
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
        "librosa": "librosa>=0.10",
        "omegaconf": "omegaconf>=2.1",
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
    print("Repository ready. Local dataset mode is enforced; no HF token is used.")
except Exception as exc:
    SETUP_EXIT_CODE = 1
    message = f"Setup failed: {type(exc).__name__}: {exc}\\n"
    print(message, end="")
    with (LOG_DIR / "setup.log").open("a", encoding="utf-8") as log:
        log.write(message)
"""),
        code("""# Configure this session. For a later session, attach the preceding Kaggle output.
LOCAL_DATASET_DIR = None  # Optional explicit root containing train/validation/test.
ATTACHED_DATASET_SOURCE = "diyalibiswas/vaani-bengali-four-dialect-audio"
PRIOR_RUN_DIR = None  # Example: Path("/kaggle/input/my-checkpoints/local-four-dialect-run")
EXPERIMENT = "moe"
CONFIG_EXIT_CODE = 0

if PRIOR_RUN_DIR:
    prior = Path(PRIOR_RUN_DIR)
    if not prior.is_dir():
        CONFIG_EXIT_CODE = 1
        print(f"Prior run directory does not exist: {prior}")
    else:
        shutil.copytree(prior, RUN_DIR, dirs_exist_ok=True)
        print(f"Restored prior checkpoints from {prior}")

print("Schedule: 3 phases x 1000 optimizer steps; progress=200; checkpoint=100")
"""),
        code("""# Locate and strictly validate the attached WAV/TXT dataset.
from collections import Counter

DISTRICT_TO_DIALECT = {
    "Alipurduar": "Kamrupi",
    "CoochBehar": "Kamrupi",
    "Darjeeling": "Kamrupi",
    "Jalpaiguri": "Kamrupi",
    "Jhargram": "Jharkhandi",
    "PaschimMedinipur": "Jharkhandi",
    "Purulia": "Jharkhandi",
    "Malda": "Varendri",
    "DakshinDinajpur": "Varendri",
    "North24Parganas": "Rarhi",
    "Kolkata": "Rarhi",
}
SPLITS = ("train", "validation", "test")
input_root = Path("/kaggle/input")

def is_dataset_root(path):
    return all((path / split).is_dir() for split in SPLITS) and all(
        (path / "train" / district).is_dir() for district in DISTRICT_TO_DIALECT
    )

candidates = sorted({path.parent for path in input_root.rglob("train") if path.is_dir() and is_dataset_root(path.parent)})
data_root = Path(LOCAL_DATASET_DIR) if LOCAL_DATASET_DIR else None
expected_slug = ATTACHED_DATASET_SOURCE.rsplit("/", 1)[-1].lower()
preferred_candidates = [path for path in candidates if expected_slug in str(path).lower()]
if data_root is None and len(preferred_candidates) == 1:
    data_root = preferred_candidates[0]
elif data_root is None and len(candidates) == 1:
    data_root = candidates[0]
elif data_root is None:
    CONFIG_EXIT_CODE = 1
    print(
        f"Could not uniquely locate {ATTACHED_DATASET_SOURCE}; "
        f"found {len(candidates)} valid roots: {candidates}"
    )

selection = {
    "kaggle_dataset_source": ATTACHED_DATASET_SOURCE,
    "district_to_dialect": DISTRICT_TO_DIALECT,
    "splits": {},
}
split_ids = {}
if data_root is not None and CONFIG_EXIT_CODE == 0:
    if not is_dataset_root(data_root):
        CONFIG_EXIT_CODE = 1
        print(f"Invalid dataset root: {data_root}")
    else:
        for split in SPLITS:
            split_dir = data_root / split
            missing_dirs = [d for d in DISTRICT_TO_DIALECT if not (split_dir / d).is_dir()]
            ignored_dirs = sorted(p.name for p in split_dir.iterdir() if p.is_dir() and p.name not in DISTRICT_TO_DIALECT)
            counts = {}
            dialect_counts = Counter()
            ids = set()
            pair_errors = []
            for district, dialect in DISTRICT_TO_DIALECT.items():
                district_dir = split_dir / district
                wav_ids = {p.relative_to(district_dir).with_suffix("") for p in district_dir.rglob("*.wav")}
                txt_ids = {p.relative_to(district_dir).with_suffix("") for p in district_dir.rglob("*.txt")}
                if wav_ids != txt_ids:
                    pair_errors.append(f"{district}: wav_only={len(wav_ids-txt_ids)}, txt_only={len(txt_ids-wav_ids)}")
                counts[district] = len(wav_ids)
                dialect_counts[dialect] += len(wav_ids)
                ids.update(f"{district}/{item.as_posix()}" for item in wav_ids)
            if missing_dirs or pair_errors or not ids:
                CONFIG_EXIT_CODE = 1
                print(f"{split} validation failed: missing={missing_dirs}, pairs={pair_errors}, samples={len(ids)}")
            split_ids[split] = ids
            selection["splits"][split] = {
                "samples": len(ids), "district_counts": counts,
                "dialect_counts": dict(dialect_counts), "ignored_districts": ignored_dirs,
            }
            print(f"{split}: selected={len(ids)} dialects={dict(dialect_counts)} ignored={ignored_dirs}")

        overlaps = {
            "train_validation": len(split_ids["train"] & split_ids["validation"]),
            "train_test": len(split_ids["train"] & split_ids["test"]),
            "validation_test": len(split_ids["validation"] & split_ids["test"]),
        }
        selection["sample_id_overlap"] = overlaps
        if any(overlaps.values()):
            CONFIG_EXIT_CODE = 1
            print(f"Cross-split sample ID overlap: {overlaps}")

if data_root is not None and CONFIG_EXIT_CODE == 0:
    selection["root"] = str(data_root)
    os.environ["VAANI_AUDIO_ROOT"] = str(data_root)
    os.environ["VAANI_ALLOW_HF_FALLBACK"] = "0"
    os.environ.pop("VAANI_PARQUET_CACHE", None)
    os.environ.pop("VAANI_LOCAL_CONFIG", None)
    (RUN_DIR / "dataset_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Using attached local dataset: {data_root}")
    print("Selected dialect labels:", sorted(set(DISTRICT_TO_DIALECT.values())))
"""),
        code("""# Train immediately with scripts/trainer.py. There is no preliminary test run.
TRAIN_EXIT_CODE = None
if SETUP_EXIT_CODE == 0 and CONFIG_EXIT_CODE == 0:
    command = [
        "accelerate", "launch",
        "--config_file", str(REPO_DIR / "configs/accelerate_t4x2.yaml"),
        str(REPO_DIR / "scripts/trainer.py"),
        "--config", str(REPO_DIR / "configs/local_four_dialect.yaml"),
        "--output-dir", str(RUN_DIR),
        "--experiment", EXPERIMENT,
        "--require-two-gpus",
    ]
    valid_checkpoints = [
        path for path in RUN_DIR.glob("checkpoint-*")
        if path.is_dir() and (path / "trainer_state.json").exists()
    ]
    if valid_checkpoints:
        command += ["--resume", "latest"]
    TRAIN_EXIT_CODE = run_logged(command, "training.log")
else:
    print("Training did not start because setup or dataset validation failed.")
print(f"training_exit_code={TRAIN_EXIT_CODE}")
"""),
        code("""# Save configuration, status, and logs as Kaggle output artifacts.
import datetime

final_checkpoint = RUN_DIR / "checkpoint-phase-3"
checkpoints = sorted(path for path in RUN_DIR.glob("checkpoint-*") if path.is_dir())
try:
    repo_commit = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
except Exception:
    repo_commit = "unavailable"

manifest = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repository_commit": repo_commit,
    "experiment": EXPERIMENT,
    "kaggle_dataset_source": ATTACHED_DATASET_SOURCE,
    "dataset_mode": "attached-local-wav-txt-only",
    "selected_districts": list(DISTRICT_TO_DIALECT),
    "dialects": sorted(set(DISTRICT_TO_DIALECT.values())),
    "steps_per_phase": 1000,
    "progress_every_steps": 200,
    "checkpoint_every_steps": 100,
    "setup_exit_code": SETUP_EXIT_CODE,
    "config_exit_code": CONFIG_EXIT_CODE,
    "training_exit_code": TRAIN_EXIT_CODE,
    "complete": final_checkpoint.exists(),
    "checkpoints": [path.name for path in checkpoints],
    "logs": [path.name for path in sorted(LOG_DIR.glob("*.log"))],
}
(RUN_DIR / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
config_path = REPO_DIR / "configs/local_four_dialect.yaml"
if config_path.exists():
    shutil.copy2(config_path, RUN_DIR / "effective_config.yaml")
logs_archive = shutil.make_archive("/kaggle/working/local-four-dialect-logs", "zip", LOG_DIR)
print(json.dumps(manifest, indent=2))
print(f"Run output: {RUN_DIR}")
print(f"Downloadable log archive: {logs_archive}")

if SETUP_EXIT_CODE != 0:
    raise RuntimeError(f"Setup failed; inspect {LOG_DIR / 'setup.log'}")
if CONFIG_EXIT_CODE != 0:
    raise RuntimeError(f"Dataset validation failed; inspect the cell output and {LOG_DIR / 'setup.log'}")
if TRAIN_EXIT_CODE not in (0, None):
    raise RuntimeError(f"Training failed with exit code {TRAIN_EXIT_CODE}; inspect {LOG_DIR / 'training.log'}")
if final_checkpoint.exists():
    print(f"All three phases completed: {final_checkpoint}")
else:
    print("Session ended before all phases completed. Save RUN_DIR as a private Dataset and resume next session.")
"""),
        markdown("""## Outputs and resuming

Kaggle persists `/kaggle/working/local-four-dialect-run` and `/kaggle/working/local-four-dialect-logs.zip` when you save a notebook version. If the three phases do not fit in one session, turn the saved run folder into a private Kaggle Dataset, attach it to the next session, set `PRIOR_RUN_DIR`, and run all cells again. The trainer resumes from the latest complete 100-step checkpoint.
"""),
    ],
}

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"local-four-{index:02d}"

root = Path(__file__).resolve().parents[1]
(root / "kaggle_local_four_dialect_training.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
)
print("Generated kaggle_local_four_dialect_training.ipynb")
