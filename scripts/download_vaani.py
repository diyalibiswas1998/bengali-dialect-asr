#!/usr/bin/env python3
"""
download_vaani.py
=================
Downloads all 11 West Bengal district configurations from the
``ARTPARK-IISc/Vaani`` HuggingFace dataset and writes JSONL manifest files
(train / validation / test) to ``data/vaani/`` so that the existing
``BengaliDialectDataset`` (JSONL-based) can also be used without change.

Audio files are saved under ``data/vaani/audio/<config>/``.

Usage
-----
  # From the project root (research/code/):
  python scripts/download_vaani.py

  # Specify a custom output directory:
  python scripts/download_vaani.py --output_dir data/vaani

  # Use only a subset of districts:
  python scripts/download_vaani.py --configs WestBengal_Kolkata WestBengal_Darjeeling

  # Download without saving audio (metadata manifests only – useful for
  # inspecting column names before a full download):
  python scripts/download_vaani.py --metadata_only

Environment
-----------
Set the environment variable ``HF_TOKEN`` (or pass ``--hf_token``) with
your HuggingFace access token.

  $env:HF_TOKEN="your_hf_token_here"
  python scripts/download_vaani.py
"""

import argparse
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_dialect_benchmark.common.constants import VAANI_DISTRICT_CONFIGS

# ──────────────────────────────────────────────────────────────────────────────
# Default token (override via --hf_token or HF_TOKEN env var)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", "")

SPLITS = ["train", "validation", "test"]
DATASET_REPO = "ARTPARK-IISc/Vaani"
SAMPLE_RATE = 16_000


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform.astype(np.float32)
    new_length = int(len(waveform) * target_sr / float(orig_sr))
    return np.interp(
        np.linspace(0, len(waveform) - 1, new_length),
        np.arange(len(waveform)),
        waveform,
    ).astype(np.float32)


def _normalize_audio(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    if waveform.dtype != np.float32 or waveform.max() > 1.5:
        waveform = waveform.astype(np.float64) / (np.iinfo(np.int16).max + 1)
    return waveform.astype(np.float32)


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a float32 waveform as a 16-bit PCM .wav file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def download_config(
    config: str,
    output_dir: Path,
    split: str,
    metadata_only: bool = False,
) -> list:
    """
    Download one (config, split) pair and return a list of JSONL record dicts.
    If *metadata_only* is True, audio is not saved to disk (audio_path will be
    an empty string).
    """
    from datasets import load_dataset

    print(f"  [{split}] {config} ...", end=" ", flush=True)
    try:
        hf_ds = load_dataset(
            DATASET_REPO,
            config,
            split=split,
            streaming=False,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return []

    audio_dir = output_dir / "audio" / config / split
    if not metadata_only:
        audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, row in enumerate(hf_ds):
        # ---- transcript ----
        transcript = (
            row.get("transcription")
            or row.get("transcript")
            or row.get("text")
            or ""
        )

        # ---- audio ----
        audio_path_str = ""
        if not metadata_only:
            audio_col = row.get("audio")
            if audio_col is not None:
                array = audio_col.get("array")
                sr = audio_col.get("sampling_rate", SAMPLE_RATE)
                if array is not None:
                    waveform = _normalize_audio(np.asarray(array))
                    waveform = _resample(waveform, sr, SAMPLE_RATE)
                    wav_path = audio_dir / f"{i:06d}.wav"
                    _write_wav(wav_path, waveform)
                    audio_path_str = str(wav_path)

        records.append(
            {
                "audio_path": audio_path_str,
                "transcript": transcript,
                "dialect_label": config,  # district config name as label
                "config": config,
                # Forward any extra scalar metadata that might be useful
                "speaker_id": str(row.get("speaker_id", "")),
                "district": str(row.get("district", "")),
            }
        )

    print(f"{len(records)} samples")
    return records


def write_manifest(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"    -> Manifest: {path}  ({len(records)} lines)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ARTPARK-IISc/Vaani West Bengal configs and generate JSONL manifests."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=VAANI_DISTRICT_CONFIGS,
        help="District configs to download (default: all 11 West Bengal districts).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        help="Splits to download (default: train validation test).",
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "data" / "vaani"),
        help="Directory to save audio files and JSONL manifests.",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN", DEFAULT_HF_TOKEN),
        help="HuggingFace access token.",
    )
    parser.add_argument(
        "--metadata_only",
        action="store_true",
        help="Write manifests without downloading audio (empty audio_path).",
    )
    args = parser.parse_args()

    # ---- HF login ----
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=args.hf_token, add_to_git_credential=False)
        print(f"Logged in to HuggingFace.")
    except Exception as exc:
        print(f"[WARNING] HuggingFace login failed: {exc}. Continuing anonymously.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"\n=== Split: {split} ===")
        all_records = []
        for config in args.configs:
            records = download_config(config, output_dir, split, args.metadata_only)
            all_records.extend(records)

        manifest_path = output_dir / f"{split}.jsonl"
        write_manifest(manifest_path, all_records)

    print("\nDone!")
    print(f"Manifests written to: {output_dir}")
    print("Update configs/config.yaml to point to the new manifests, e.g.:")
    print(f"  data:")
    print(f"    train_manifest: data/vaani/train.jsonl")
    print(f"    val_manifest:   data/vaani/validation.jsonl")
    print(f"    test_manifest:  data/vaani/test.jsonl")


if __name__ == "__main__":
    main()
