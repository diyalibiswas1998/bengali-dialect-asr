#!/usr/bin/env python
"""Audit CTC wiring, labels, lengths, logits, and decoding for one checkpoint.

This script is read-only with respect to the dataset and checkpoint.  It writes
JSON/CSV reports below ``--output-dir`` and never starts training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

EXPECTED_BLANK_ID = 0
EXPECTED_DELIMITER_ID = 2
EXPECTED_VOCAB_SIZE = 73
TARGET_SAMPLE_RATE = 16_000

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

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = ZERO_WIDTH_RE.sub("", text)
    cleaned = []
    for character in text:
        if character.isspace():
            cleaned.append(" ")
        elif "\u0980" <= character <= "\u09FF" and unicodedata.category(character)[0] in {"L", "M", "N"}:
            cleaned.append(character)
        else:
            cleaned.append(" ")
    return WHITESPACE_RE.sub(" ", "".join(cleaned)).strip()


def split_layout(root: Path) -> str:
    if all((root / split).is_dir() for split in ("train", "validation", "test")):
        return "directories"
    if all((root / f"{split}.zip").is_file() for split in ("train", "validation", "test")):
        return "split-zips"
    raise RuntimeError(
        f"{root} must contain train/validation/test directories or split ZIP files"
    )


def iter_pairs(root: Path, split: str) -> Iterator[dict]:
    """Yield matched WAV/TXT records for exactly the 11 approved districts."""
    mode = split_layout(root)
    if mode == "directories":
        split_dir = root / split
        for district in DISTRICT_TO_DIALECT:
            district_dir = split_dir / district
            if not district_dir.is_dir():
                raise RuntimeError(f"Missing required district directory: {district_dir}")
            txt_by_key = {
                path.relative_to(district_dir).with_suffix("").as_posix(): path
                for path in district_dir.rglob("*.txt")
            }
            wav_by_key = {
                path.relative_to(district_dir).with_suffix("").as_posix(): path
                for path in district_dir.rglob("*.wav")
            }
            if set(txt_by_key) != set(wav_by_key):
                raise RuntimeError(
                    f"WAV/TXT mismatch in {district}: "
                    f"missing_audio={len(set(txt_by_key) - set(wav_by_key))} "
                    f"missing_text={len(set(wav_by_key) - set(txt_by_key))}"
                )
            for key in sorted(txt_by_key):
                transcript = normalize_text(
                    txt_by_key[key].read_text(encoding="utf-8-sig", errors="strict")
                )
                if transcript:
                    yield {
                        "sample_id": f"{district}/{key}",
                        "audio": wav_by_key[key],
                        "transcript": transcript,
                        "district": district,
                        "dialect": DISTRICT_TO_DIALECT[district],
                    }
        return

    with zipfile.ZipFile(root / f"{split}.zip") as archive:
        grouped: dict[tuple[str, str], dict[str, str]] = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = list(PurePosixPath(info.filename).parts)
            if parts and parts[0].lower() == split.lower():
                parts = parts[1:]
            if len(parts) < 2 or parts[0] not in DISTRICT_TO_DIALECT:
                continue
            relative = PurePosixPath(*parts[1:])
            suffix = relative.suffix.lower()
            if suffix not in {".wav", ".txt"}:
                continue
            grouped.setdefault((parts[0], relative.with_suffix("").as_posix()), {})[
                suffix
            ] = info.filename
        for (district, key), pair in sorted(grouped.items()):
            if set(pair) != {".wav", ".txt"}:
                raise RuntimeError(f"ZIP WAV/TXT mismatch: {district}/{key}")
            transcript = normalize_text(
                archive.read(pair[".txt"]).decode("utf-8-sig", errors="strict")
            )
            if transcript:
                yield {
                    "sample_id": f"{district}/{key}",
                    "audio": (root / f"{split}.zip", pair[".wav"]),
                    "transcript": transcript,
                    "district": district,
                    "dialect": DISTRICT_TO_DIALECT[district],
                }


def choose_rows(root: Path, split: str, limit: int, seed: int = 42) -> list[dict]:
    rows = list(iter_pairs(root, split))
    rows.sort(key=lambda row: row["sample_id"])
    if limit <= 0 or len(rows) <= limit:
        return rows
    groups = {district: [] for district in DISTRICT_TO_DIALECT}
    for row in rows:
        groups[row["district"]].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < limit:
        progressed = False
        for district in DISTRICT_TO_DIALECT:
            if groups[district] and len(selected) < limit:
                selected.append(groups[district].pop())
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: row["sample_id"])
    return selected


def make_manifest(data_root: Path, output: Path, count: int = 32, seed: int = 42) -> dict:
    rows = choose_rows(data_root, "test", max(count, len(DISTRICT_TO_DIALECT)), seed)
    rows = rows[:count]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "audio", "transcript", "district", "dialect", "manually_verified"],
        )
        writer.writeheader()
        for row in rows:
            audio = row["audio"]
            if isinstance(audio, tuple):
                audio = f"{audio[0]}::{audio[1]}"
            writer.writerow({**row, "audio": str(audio), "manually_verified": "NO"})
    summary = {
        "manifest": str(output),
        "count": len(rows),
        "district_counts": dict(Counter(row["district"] for row in rows)),
        "requires_manual_audio_transcript_check": True,
    }
    output.with_name("tiny_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def read_audio(source, archive_cache: dict | None = None):
    import io
    import soundfile as sf
    import torch
    import torchaudio

    if isinstance(source, tuple):
        archive_cache = archive_cache if archive_cache is not None else {}
        archive = archive_cache.get(str(source[0]))
        if archive is None:
            archive = zipfile.ZipFile(source[0])
            archive_cache[str(source[0])] = archive
        handle = io.BytesIO(archive.read(source[1]))
    else:
        handle = source
    waveform, sample_rate = sf.read(handle, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Audio is empty or contains non-finite samples")
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform), int(sample_rate), TARGET_SAMPLE_RATE
        ).numpy()
    return waveform.astype("float32", copy=False)


def _load_processor(checkpoint: Path, model_name: str):
    from transformers import Wav2Vec2Processor

    candidate = checkpoint / "processor"
    if not candidate.is_dir():
        candidate = checkpoint
    processor = Wav2Vec2Processor.from_pretrained(candidate)
    tokenizer = processor.tokenizer
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Expected Bengali CTC vocabulary {EXPECTED_VOCAB_SIZE}, got {len(tokenizer)}")
    if tokenizer.pad_token_id != EXPECTED_BLANK_ID:
        raise ValueError(f"Expected blank/pad ID 0, got {tokenizer.pad_token_id}")
    if tokenizer.unk_token_id != 1:
        raise ValueError(f"Expected unknown ID 1, got {tokenizer.unk_token_id}")
    if tokenizer.convert_tokens_to_ids("|") != EXPECTED_DELIMITER_ID:
        raise ValueError("Expected word delimiter ID 2")
    if tokenizer.word_delimiter_token != "|":
        raise ValueError("Expected word delimiter token '|'")
    feature = processor.feature_extractor
    if int(feature.sampling_rate) != TARGET_SAMPLE_RATE or not bool(feature.do_normalize):
        raise ValueError("Checkpoint processor is not the normalized 16-kHz MMS processor")
    return processor


def _load_model(checkpoint: Path, repo_root: Path, processor, model_name: str, device):
    import torch
    from omegaconf import OmegaConf
    from asr_dialect_benchmark.modeling import BengaliDialectASR

    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    saved_tokens = int(config_data.get("model", {}).get("num_tokens", EXPECTED_VOCAB_SIZE))
    if saved_tokens != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            "Checkpoint model.num_tokens is not the Bengali CTC size: "
            f"{saved_tokens}"
        )
    config = OmegaConf.create(config_data)
    config.model.num_tokens = EXPECTED_VOCAB_SIZE
    config.model.gradient_checkpointing = False
    model = BengaliDialectASR(config)
    if model.ctc_head.out_features != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"CTC head has {model.ctc_head.out_features} outputs")

    state_path = None
    for candidate in ("model.safetensors", "model_state.pt", "pytorch_model.bin"):
        if (checkpoint / candidate).is_file():
            state_path = checkpoint / candidate
            break
    if state_path is None:
        raise FileNotFoundError(f"No model state found in {checkpoint}")
    if state_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(state_path), device="cpu")
    else:
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    state = {
        (key.removeprefix("module.") if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any(key.startswith("ctc_head.") for key in missing):
        raise ValueError(f"Checkpoint is missing Bengali CTC head weights: {missing}")
    model.to(device).eval()
    return model, config_data, {
        "state_path": str(state_path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _decode(processor, ids: Iterable[int], blank_id: int) -> str:
    collapsed = []
    previous = None
    for raw in ids:
        token_id = int(raw)
        if token_id == blank_id:
            previous = token_id
            continue
        if token_id == previous:
            continue
        collapsed.append(token_id)
        previous = token_id
    return processor.tokenizer.decode(
        collapsed, group_tokens=False, skip_special_tokens=True
    ).strip()


def edit_distance(reference: list, hypothesis: list) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i]
        for j, hyp in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def _bias_report(model) -> dict:
    bias = model.ctc_head.bias.detach().float().cpu()
    values = bias.tolist()
    largest = sorted(enumerate(values), key=lambda item: item[1], reverse=True)[:10]
    return {
        "blank_bias_id_0": float(values[EXPECTED_BLANK_ID]),
        "delimiter_bias_id_2": float(values[EXPECTED_DELIMITER_ID]),
        "largest_biases": [[int(index), float(value)] for index, value in largest],
        "initialization": "torch.nn.Linear default initialization unless checkpoint metadata says otherwise",
    }


def audit_checkpoint(
    checkpoint: Path,
    data_root: Path,
    repo_root: Path,
    output_dir: Path,
    sample_count: int,
    batch_size: int,
    model_name: str,
    seed: int,
) -> dict:
    import torch

    sys.path.insert(0, str(repo_root / "src"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = _load_processor(checkpoint, model_name)
    model, saved_config, state_report = _load_model(
        checkpoint, repo_root, processor, model_name, device
    )
    rows = choose_rows(data_root, "validation", sample_count, seed)
    if not rows:
        raise RuntimeError("No validation records were selected")

    label_counts = Counter()
    label_total = 0
    label_examples = []
    frame_counts = Counter()
    predictions = []
    invalid_lengths = []
    blank_probabilities = []
    archive_cache = {}
    for start in range(0, len(rows), max(1, batch_size)):
        batch_rows = rows[start : start + max(1, batch_size)]
        arrays = [torch.from_numpy(read_audio(row["audio"], archive_cache)) for row in batch_rows]
        processed = processor.feature_extractor(
            [array.numpy() for array in arrays],
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = processed["input_values"].to(device)
        attention_mask = processed["attention_mask"].to(device)
        input_lengths = attention_mask.sum(-1).long()
        target_lists = [processor.tokenizer(row["transcript"]).input_ids for row in batch_rows]
        for row, target in zip(batch_rows, target_lists):
            label_counts.update(int(item) for item in target)
            label_total += len(target)
            if len(label_examples) < 5:
                label_examples.append(
                    {
                        "sample_id": row["sample_id"],
                        "reference": row["transcript"],
                        "target_ids": target,
                        "decoded_target": processor.tokenizer.decode(
                            target, group_tokens=False, skip_special_tokens=True
                        ).strip(),
                    }
                )
        with torch.inference_mode():
            outputs = model(input_values, attention_mask, input_lengths)
        logits = outputs["logits"].float()
        assert logits.ndim == 3
        assert logits.shape[-1] == EXPECTED_VOCAB_SIZE
        assert int(processor.tokenizer.pad_token_id) == EXPECTED_BLANK_ID
        output_lengths = outputs["output_lengths"].long().clamp(0, logits.shape[1])
        ids = logits.argmax(-1)
        probabilities = logits.softmax(-1)
        for row_index, row in enumerate(batch_rows):
            length = int(output_lengths[row_index].item())
            sequence = ids[row_index, :length].detach().cpu().tolist()
            frame_counts.update(sequence)
            target = target_lists[row_index]
            repeats = sum(left == right for left, right in zip(target, target[1:]))
            minimum = len(target) + repeats
            if length < minimum:
                invalid_lengths.append(
                    {
                        "sample_id": row["sample_id"],
                        "output_length": length,
                        "target_length": len(target),
                        "adjacent_repeats": repeats,
                        "minimum_required": minimum,
                    }
                )
            probability = probabilities[row_index, :length]
            blank_probabilities.append(float(probability[:, EXPECTED_BLANK_ID].mean().item()))
            prediction = _decode(processor, sequence, EXPECTED_BLANK_ID)
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "district": row["district"],
                    "dialect": row["dialect"],
                    "reference": row["transcript"],
                    "prediction": prediction,
                    "empty_prediction": not bool(prediction),
                    "output_length": length,
                    "target_length": len(target),
                    "top_frame_token_ids": Counter(sequence).most_common(15),
                }
            )

    label_top = [[int(index), int(count)] for index, count in label_counts.most_common(20)]
    frame_total = max(1, sum(frame_counts.values()))
    reference_chars = sum(len(row["reference"].replace(" ", "")) for row in predictions)
    prediction_chars = sum(len(row["prediction"].replace(" ", "")) for row in predictions)
    cer_distance = sum(
        edit_distance(list(row["reference"].replace(" ", "")), list(row["prediction"].replace(" ", "")))
        for row in predictions
    )
    wer_distance = sum(
        edit_distance(row["reference"].split(), row["prediction"].split())
        for row in predictions
    )
    reference_words = sum(len(row["reference"].split()) for row in predictions)
    report = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "device": str(device),
        "model_path": state_report,
        "saved_model_num_tokens": int(saved_config.get("model", {}).get("num_tokens", -1)),
        "ctc_contract": {
            "blank_id": EXPECTED_BLANK_ID,
            "padding_id": int(processor.tokenizer.pad_token_id),
            "unknown_id": int(processor.tokenizer.unk_token_id),
            "delimiter_id": EXPECTED_DELIMITER_ID,
            "vocabulary_size": EXPECTED_VOCAB_SIZE,
            "ctc_head_out_features": int(model.ctc_head.out_features),
            "feature_sampling_rate": int(processor.feature_extractor.sampling_rate),
            "feature_do_normalize": bool(processor.feature_extractor.do_normalize),
            "tensor_path": "MMS encoder -> optional MoE -> ctc_head(73) -> logits -> log_softmax -> CTCLoss(blank=0)",
        },
        "label_audit": {
            "valid_target_tokens": label_total,
            "blank_id_0_count": int(label_counts[EXPECTED_BLANK_ID]),
            "delimiter_id_2_count": int(label_counts[EXPECTED_DELIMITER_ID]),
            "unknown_id_1_count": int(label_counts[1]),
            "blank_fraction": label_counts[EXPECTED_BLANK_ID] / max(1, label_total),
            "delimiter_fraction": label_counts[EXPECTED_DELIMITER_ID] / max(1, label_total),
            "unknown_fraction": label_counts[1] / max(1, label_total),
            "top_20_target_ids": label_top,
            "decoded_examples": label_examples,
        },
        "length_audit": {
            "sample_count": len(rows),
            "invalid_count": len(invalid_lengths),
            "invalid_samples": invalid_lengths[:50],
            "zero_infinity_would_hide_invalid_samples": bool(invalid_lengths),
        },
        "raw_prediction_audit": {
            "frame_count": int(frame_total),
            "blank_argmax_fraction": frame_counts[EXPECTED_BLANK_ID] / frame_total,
            "delimiter_argmax_fraction": frame_counts[EXPECTED_DELIMITER_ID] / frame_total,
            "blank_mean_probability": sum(blank_probabilities) / max(1, len(blank_probabilities)),
            "empty_prediction_rate": sum(item["empty_prediction"] for item in predictions) / max(1, len(predictions)),
            "top_15_frame_token_ids": [[int(index), int(count)] for index, count in frame_counts.most_common(15)],
            "cer": cer_distance / max(1, reference_chars),
            "wer": wer_distance / max(1, reference_words),
            "predictions": predictions,
        },
        "ctc_head_bias": _bias_report(model),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint.name)
    (output_dir / f"ctc_collapse_{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / f"ctc_predictions_{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-name", default="facebook/mms-300m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make-manifest", type=Path)
    parser.add_argument("--manifest-count", type=int, default=32)
    args = parser.parse_args()
    if args.make_manifest:
        print(json.dumps(make_manifest(args.data_root, args.make_manifest, args.manifest_count, args.seed), indent=2))
    if not args.checkpoint:
        if args.make_manifest:
            return
        parser.error("--checkpoint is required unless --make-manifest is used")
    reports = [
        audit_checkpoint(
            checkpoint.resolve(),
            args.data_root.resolve(),
            args.repo_root.resolve(),
            args.output_dir.resolve(),
            args.sample_count,
            args.batch_size,
            args.model_name,
            args.seed,
        )
        for checkpoint in args.checkpoint
    ]
    summary = {"checkpoints": reports}
    (args.output_dir / "ctc_collapse_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
