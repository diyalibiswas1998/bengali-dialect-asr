"""Generate the Kaggle notebook that creates the processed private dataset."""

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
        "kaggle": {"accelerator": "none", "isInternetEnabled": True},
    },
    "cells": [
        markdown("""# Build the private processed Bengali Vaani dataset

Run this once with Internet enabled. Add `HF_TOKEN` as a Kaggle secret after accepting the upstream dataset terms. The token is read from the secret store and is never written to an artifact. The builder first inspects the transcribed Bengali subset and fails safely if required metadata is absent. Set `ALLOW_MAIN_FALLBACK=True` only after explicitly accepting the much larger 11-district raw-corpus build.
"""),
        code("""import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/diyalibiswas1998/bengali-dialect-asr.git"
REPO_DIR = Path("/kaggle/working/bengali-dialect-asr")
if not REPO_DIR.exists():
    subprocess.check_call(["git", "clone", REPO_URL, str(REPO_DIR)])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)])

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    raise RuntimeError("Add an HF_TOKEN Kaggle secret after accepting the Vaani dataset terms") from exc
print("Repository and dependencies are ready; the token value was not displayed.")
"""),
        code("""import subprocess, sys
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/vaani-bengali-processed")
ALLOW_MAIN_FALLBACK = False
RESUME_STAGING = None  # Set to a preserved /kaggle/working/.vaani-bengali-processed-building-* directory.
if OUTPUT_DIR.exists():
    raise FileExistsError(f"{OUTPUT_DIR} already exists. Remove it only if you intend to restart the build.")
command = [
    sys.executable, str(REPO_DIR / "scripts/build_processed_vaani.py"),
    "--source", "auto",
    "--output-dir", str(OUTPUT_DIR),
]
if ALLOW_MAIN_FALLBACK:
    command.append("--allow-main-fallback")
if RESUME_STAGING:
    command += ["--resume-staging", str(RESUME_STAGING)]
subprocess.check_call(command)
"""),
        code("""subprocess.check_call([
    sys.executable, str(REPO_DIR / "scripts/validate_processed.py"),
    "--data-dir", str(OUTPUT_DIR),
    "--decode-audio",
])
print("Validation passed. Use Kaggle's Save Version / Create Dataset flow to publish OUTPUT_DIR as a private dataset.")
"""),
        markdown("""## Publication checklist

- Keep the Kaggle Dataset private unless upstream redistribution terms clearly allow publication.
- Include `SOURCE_AND_LICENSE.md`, `metadata.json`, `vocab.json`, and `dialect_mapping.json`.
- Record the Kaggle Dataset version used by every experiment.
- Do not claim the four geographic proxy groups are definitive dialect ground truth without linguistic review.
"""),
    ],
}

code_root = Path(__file__).resolve().parents[1]
for destination in (code_root / "kaggle_dataset_creator.ipynb", code_root.parent / "kaggle_dataset_creator.ipynb"):
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("Generated Kaggle dataset-creator notebooks.")
