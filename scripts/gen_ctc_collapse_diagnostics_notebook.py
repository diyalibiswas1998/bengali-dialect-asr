#!/usr/bin/env python
"""Generate a Kaggle notebook that audits CTC collapse without full training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def code(source: str):
    return new_code_cell(source.strip() + "\n")


def build(commit: str) -> nbformat.NotebookNode:
    intro = f"""# Bengali MMS CTC collapse diagnostics

This notebook is diagnostic-only. It does **not** start the 6,000-step MoE
training job. It checks a saved checkpoint against the exact Bengali CTC
contract: vocabulary 73, blank/pad ID 0, unknown ID 1, and word delimiter ID 2.

Repository commit: `{commit}`

Attach both the balanced four-dialect dataset and the Kaggle Dataset containing
the saved checkpoint before running the audit cell. The checkpoint dataset must
contain `config.json`, `model.safetensors` (or `model_state.pt`), and the saved
processor files.
"""
    setup = f"""
from pathlib import Path
import json, os, shutil, subprocess, sys, time, zipfile

REPO_URL = "https://github.com/diyalibiswas1998/bengali-dialect-asr.git"
REPO_COMMIT = "{commit}"
REPO_DIR = Path("/kaggle/working/bengali-dialect-asr")
RUN_DIR = Path("/kaggle/working/ctc-collapse-diagnostics")
LOG_DIR = RUN_DIR / "logs"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--all", "--tags"], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", REPO_COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")], check=True)
print("repo_commit=" + subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip())
print("No HF token is required; the MMS model is public. Internet must be enabled for the first model download.")

INPUT_ROOT = Path("/kaggle/input")
DATASET_SLUG = "four-dialect-data-undersampled"
def is_dataset_root(path: Path) -> bool:
    return all((path / split).is_dir() or (path / f"{{split}}.zip").is_file() for split in ("train", "validation", "test"))

dataset_candidates = [path for path in INPUT_ROOT.rglob("*") if path.is_dir() and DATASET_SLUG in str(path).lower() and is_dataset_root(path)]
if not dataset_candidates:
    dataset_candidates = [path for path in INPUT_ROOT.rglob("*") if path.is_dir() and is_dataset_root(path)]
if not dataset_candidates:
    raise FileNotFoundError("Attach diyalibiswas/four-dialect-data-undersampled to the notebook input")
DATA_ROOT = sorted(dataset_candidates, key=lambda path: len(str(path)))[0]
print("DATA_ROOT=", DATA_ROOT)

CHECKPOINT_OVERRIDE = os.environ.get("CTC_CHECKPOINT", "").strip()
CHECKPOINT_ROOT = Path(CHECKPOINT_OVERRIDE) if CHECKPOINT_OVERRIDE else None
if CHECKPOINT_ROOT and not CHECKPOINT_ROOT.exists():
    raise FileNotFoundError(CHECKPOINT_ROOT)
if CHECKPOINT_ROOT is None:
    checkpoint_candidates = []
    for config_path in INPUT_ROOT.rglob("config.json"):
        candidate = config_path.parent
        if (candidate / "model.safetensors").is_file() or (candidate / "model_state.pt").is_file() or (candidate / "pytorch_model.bin").is_file():
            checkpoint_candidates.append(candidate)
    if checkpoint_candidates:
        def checkpoint_step(path):
            state_path = path / "trainer_state.json"
            if not state_path.is_file():
                return -1
            try:
                return int(json.loads(state_path.read_text(encoding="utf-8")).get("global_step", -1))
            except Exception:
                return -1
        CHECKPOINT_ROOT = sorted(checkpoint_candidates, key=checkpoint_step)[-1]
print("CHECKPOINT_ROOT=", CHECKPOINT_ROOT)
"""
    manifest = """
# Build a deterministic 32-example manifest. Listen to every pair and change
# manually_verified from NO to YES before running the tiny overfit command.
manifest_path = RUN_DIR / "tiny_manifest.csv"
manifest_command = [
    sys.executable, str(REPO_DIR / "scripts" / "ctc_collapse_diagnostics.py"),
    "--data-root", str(DATA_ROOT), "--repo-root", str(REPO_DIR),
    "--output-dir", str(RUN_DIR), "--make-manifest", str(manifest_path),
    "--manifest-count", "32",
]
subprocess.run(manifest_command, check=True)
print(manifest_path.read_text(encoding="utf-8")[:1500])
"""
    audit = """
# Audit the latest attached checkpoint. No training is launched.
audit_log = LOG_DIR / "checkpoint_audit.log"
if CHECKPOINT_ROOT is None:
    status = {
        "status": "blocked",
        "reason": "No attached checkpoint dataset was found",
        "action": "Attach the Kaggle output dataset containing checkpoint files and rerun this cell",
    }
    (RUN_DIR / "checkpoint_audit_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
else:
    command = [
        sys.executable, str(REPO_DIR / "scripts" / "ctc_collapse_diagnostics.py"),
        "--data-root", str(DATA_ROOT), "--repo-root", str(REPO_DIR),
        "--output-dir", str(RUN_DIR), "--checkpoint", str(CHECKPOINT_ROOT),
        "--sample-count", "100", "--batch-size", "4",
    ]
    with audit_log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"Checkpoint audit failed; inspect {{audit_log}}")
    print((RUN_DIR / "ctc_collapse_summary.json").read_text(encoding="utf-8"))
"""
    tiny = """
# Tiny test command (intentionally not auto-run). It is one GPU, plain MMS-CTC,
# no MoE, no dialect loss, no augmentation, and no distributed launcher.
tiny_command = [
    sys.executable, str(REPO_DIR / "scripts" / "tiny_overfit_ctc.py"),
    "--manifest", str(manifest_path), "--checkpoint", str(CHECKPOINT_ROOT or "<attach-checkpoint>"),
    "--output-dir", str(RUN_DIR / "tiny-overfit"), "--batch-size", "4",
    "--max-steps", "3000", "--eval-every", "50", "--manually-verified",
]
print("After listening to and marking all 32 rows YES, run:")
print(" ".join(map(str, tiny_command)))
"""
    outputs = """
# Persist logs and reports as a single Kaggle output artifact.
manifest = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "repository_commit": REPO_COMMIT,
    "dataset_root": str(DATA_ROOT),
    "checkpoint_root": str(CHECKPOINT_ROOT) if CHECKPOINT_ROOT else None,
    "diagnostic_only": True,
}
(RUN_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
archive = Path("/kaggle/working/ctc-collapse-diagnostics-outputs.zip")
if archive.exists():
    archive.unlink()
shutil.make_archive(str(archive.with_suffix("")), "zip", RUN_DIR)
print("Saved:", archive)
for path in sorted(RUN_DIR.rglob("*")):
    if path.is_file():
        print(path.relative_to(RUN_DIR), path.stat().st_size)
"""
    return new_notebook(
        cells=[
            new_markdown_cell(intro),
            code(setup),
            code(manifest),
            code(audit),
            code(tiny),
            code(outputs),
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
            "modified_by": "Codex: Bengali CTC contract and collapse diagnostics",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build(args.commit, ), args.output)
    metadata = {
        "id": "diyalibiswas/bengali-ctc-collapse-diagnostics",
        "title": "Bengali CTC Collapse Diagnostics",
        "code_file": args.output.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["diyalibiswas/four-dialect-data-undersampled"],
        "machine_shape": "NvidiaTeslaT4",
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
