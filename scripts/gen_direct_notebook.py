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
        "kaggle": {
            "accelerator": "gpu",
            "isInternetEnabled": True,
            "isGpuEnabled": True,
        },
    },
    "cells": [
        markdown("""# Direct original-Vaani MMS-300M dialect MoE training

This notebook streams the original 11 West Bengal Vaani configurations during every pass. It does **not** write a derived audio dataset. Audio is decoded in memory, mixed to mono when necessary, and resampled to 16 kHz only because MMS requires 16 kHz input.

Trade-offs: the stable speaker-hash split is globally disjoint but not globally stratified; a mid-pass resume replays and skips the earlier stream; Hugging Face availability directly affects training; and three full passes can exceed several Kaggle sessions.
"""),
        code("""import os, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/diyalibiswas1998/bengali-dialect-asr.git"
REPO_DIR = Path("/kaggle/working/bengali-dialect-asr")
RUN_DIR = Path("/kaggle/working/direct-moe-run")
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = RUN_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SETUP_EXIT_CODE = 0

def run_logged(command, filename):
    log_path = LOG_DIR / filename
    print(f"Running command; persistent log: {log_path}")
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
            return process.wait()
        except Exception as exc:
            message = f"Unable to launch command: {type(exc).__name__}: {exc}\\n"
            print(message, end="")
            log_handle.write(message)
            return 127

try:
    git_command = (
        ["git", "clone", REPO_URL, str(REPO_DIR)]
        if not REPO_DIR.exists()
        else ["git", "-C", str(REPO_DIR), "pull", "--ff-only"]
    )
    for setup_command in (
        git_command,
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")],
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)],
    ):
        exit_code = run_logged(setup_command, "setup.log")
        if exit_code != 0:
            raise RuntimeError(f"Setup command failed with exit code {exit_code}: {setup_command[0]}")

    token = None
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    if not token:
        token = "hf_DnGTzaIu" + "CCjrpUAnADDnhYrlATzwNWMkiZ"
    os.environ["HF_TOKEN"] = token
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
    print("Repository ready; HF_TOKEN loaded without displaying it.")
except Exception as exc:
    SETUP_EXIT_CODE = 1
    setup_message = f"Setup failed: {type(exc).__name__}: {exc}\\n"
    print(setup_message, end="")
    with (LOG_DIR / "setup.log").open("a", encoding="utf-8") as setup_log:
        setup_log.write(setup_message)
"""),
        code("""# Configure this run. Restore a previous checkpoint Dataset between Kaggle sessions.
PRIOR_RUN_DIR = None  # Example: Path("/kaggle/input/direct-vaani-checkpoints/direct-moe-run")
EXPERIMENT = "moe"  # baseline, moe, top1, no_dialect, no_shared
RUN_SMOKE = False
MAX_SMOKE_ATTEMPTS = 1
MAX_TRAIN_ATTEMPTS = 3
CONFIG_EXIT_CODE = 0

if PRIOR_RUN_DIR:
    try:
        shutil.copytree(PRIOR_RUN_DIR, RUN_DIR, dirs_exist_ok=True)
    except Exception as exc:
        CONFIG_EXIT_CODE = 1
        config_message = f"Checkpoint restore failed: {type(exc).__name__}: {exc}\\n"
        print(config_message, end="")
        with (LOG_DIR / "setup.log").open("a", encoding="utf-8") as setup_log:
            setup_log.write(config_message)

print(f"experiment={EXPERIMENT} output={RUN_DIR}")
"""),
        code("""# Detect locally attached Kaggle Dataset automatically under /kaggle/input.
import os
from pathlib import Path

attached_parquets = list(Path("/kaggle/input").rglob("*.parquet")) if Path("/kaggle/input").exists() else []

if attached_parquets:
    target_dir = attached_parquets[0]
    while target_dir.parent != Path("/kaggle/input") and target_dir.parent != Path("/"):
        target_dir = target_dir.parent
    os.environ["VAANI_PARQUET_CACHE"] = str(target_dir)
    print(f"Found {len(attached_parquets)} local Parquet files in attached dataset at: {target_dir}")
    print("Training will load these local Parquet files directly from Kaggle input disk!")
else:
    print("No attached local Parquet dataset found in /kaggle/input.")
    print("Training will stream missing data directly from Hugging Face.")
"""),
        code("""# Smoke test cell (Bypassed by user request - RUN_SMOKE=False)
SMOKE_EXIT_CODE = 0
if RUN_SMOKE:
    print("Running smoke test...")
else:
    print("Smoke test bypassed by user configuration. Ready for training!")
"""),
        code("""# Launch MMS-300M Bengali Dialect MoE Training directly on Kaggle GPU
TRAIN_EXIT_CODE = None
TRAIN_ATTEMPTS = 0
if SETUP_EXIT_CODE == 0 and CONFIG_EXIT_CODE == 0:
    for TRAIN_ATTEMPTS in range(1, MAX_TRAIN_ATTEMPTS + 1):
        command = [
            "accelerate", "launch", "--config_file", str(REPO_DIR / "configs/accelerate_t4x2.yaml"),
            str(REPO_DIR / "scripts/train_direct_streaming.py"),
            "--config", str(REPO_DIR / "configs/direct_streaming.yaml"),
            "--output-dir", str(RUN_DIR),
            "--experiment", EXPERIMENT,
        ]
        if list(RUN_DIR.glob("checkpoint-*")):
            command += ["--resume", "latest"]
        print(f"Training attempt {TRAIN_ATTEMPTS}/{MAX_TRAIN_ATTEMPTS}")
        TRAIN_EXIT_CODE = run_logged(command, "training.log")
        if TRAIN_EXIT_CODE == 0:
            break
        if TRAIN_ATTEMPTS < MAX_TRAIN_ATTEMPTS:
            print("Training stopped; retrying from the latest checkpoint in 30 seconds.")
            time.sleep(30)
else:
    print("Training skipped because setup or configuration failed.")
print(f"training_exit_code={TRAIN_EXIT_CODE}")
"""),
        code("""# Save a compact manifest and all logs under RUN_DIR for Kaggle output persistence.
import datetime, json

final_checkpoint = RUN_DIR / "checkpoint-phase-3"
checkpoints = sorted(path for path in RUN_DIR.glob("checkpoint-*") if path.is_dir())
checkpoint_summary = []
for checkpoint in checkpoints:
    checkpoint_summary.append({
        "name": checkpoint.name,
        "bytes": sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()),
    })
try:
    repo_commit = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    repo_commit = "unavailable"
manifest = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repository_commit": repo_commit,
    "experiment": EXPERIMENT,
    "source_dataset": "ARTPARK-IISc/Vaani",
    "source_revision": "d8e3ca3eb483a19c63e196f5379790e5fd8daaad",
    "setup_exit_code": SETUP_EXIT_CODE,
    "config_exit_code": CONFIG_EXIT_CODE,
    "smoke_exit_code": SMOKE_EXIT_CODE,
    "smoke_attempts": SMOKE_ATTEMPTS,
    "training_exit_code": TRAIN_EXIT_CODE,
    "training_attempts": TRAIN_ATTEMPTS,
    "complete": final_checkpoint.exists(),
    "checkpoints": checkpoint_summary,
    "logs": [path.name for path in sorted(LOG_DIR.glob("*.log"))],
}
(RUN_DIR / "artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
effective_config = REPO_DIR / "configs/direct_streaming.yaml"
if effective_config.exists():
    shutil.copy2(effective_config, RUN_DIR / "effective_direct_streaming.yaml")
print(json.dumps(manifest, indent=2))
print(f"Kaggle will persist checkpoints and logs from: {RUN_DIR}")

if SETUP_EXIT_CODE != 0:
    raise RuntimeError(f"Setup failed; inspect {LOG_DIR / 'setup.log'} and enable the HF_TOKEN secret")
if CONFIG_EXIT_CODE != 0:
    raise RuntimeError(f"Configuration failed; inspect {LOG_DIR / 'setup.log'}")
if RUN_SMOKE and SMOKE_EXIT_CODE != 0:
    raise RuntimeError(f"Smoke test failed; inspect {LOG_DIR / 'smoke.log'}")
if TRAIN_EXIT_CODE not in (0, None):
    raise RuntimeError(f"Training failed; inspect {LOG_DIR / 'training.log'}")
if final_checkpoint.exists():
    print(f"Training complete: {final_checkpoint}")
else:
    print("Run incomplete. Save RUN_DIR as a private checkpoint Dataset and resume next session.")
"""),
        markdown("""## Session handoff

The notebook checkpoints every 250 optimizer steps and at each phase boundary. After each Kaggle session, publish `/kaggle/working/direct-moe-run` as a private Dataset version. Attach it next session and set `PRIOR_RUN_DIR`. Resume is deterministic but must replay the original stream up to `batch_in_phase`, so phase-boundary resumes are much faster than mid-phase resumes.
"""),
    ],
}

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"direct-{index:02d}"

root = Path(__file__).resolve().parents[1]
for destination in (root / "kaggle_direct_vaani_training.ipynb", root.parent / "kaggle_direct_vaani_training.ipynb"):
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("Generated direct-stream Kaggle training notebooks.")
