#!/usr/bin/env python
"""Controlled 32-sample plain MMS-CTC overfit experiment.

This module intentionally has no dependency on the Bengali MoE model or on a
training checkpoint.  It loads a fresh ``facebook/mms-300m`` backbone, keeps
the validated processor contract explicit, and records enough evidence to
decide the tiny overfit gate at one common checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
import unicodedata
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from asr_dialect_benchmark.tokenization.simple_tokenizer import normalize_bengali_text


EXPECTED_VOCAB_SIZE = 73
EXPECTED_BLANK_ID = 0
EXPECTED_UNKNOWN_ID = 1
EXPECTED_DELIMITER_ID = 2
EXPECTED_DELIMITER_TOKEN = "|"
TARGET_SAMPLE_RATE = 16_000
GATE_CER = 0.05
GATE_WER = 0.05
GATE_EMPTY_RATE = 0.10
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

LOGGER = logging.getLogger("tiny_overfit_ctc")


@dataclass
class Sample:
    sample_id: str
    audio_path: str
    transcript: str
    dialect: str = ""
    district: str = ""
    manually_verified: str = "YES"
    duration_seconds: float | None = None
    audio_sha256: str = ""
    transcript_sha256: str = ""
    raw_transcript: str = ""
    normalized_transcript: str = ""
    target_ids: tuple[int, ...] = ()
    decoded_target: str = ""
    raw_changed: bool = False
    adjacent_repeat_count: int = 0
    resolved_audio_path: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file atomically so an interrupted run cannot corrupt JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def git_commit(project_root: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / "tiny_overfit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)
    audit_handler = logging.FileHandler(output_dir / "logs/checkpoint_audit.log", encoding="utf-8")
    audit_handler.setFormatter(formatter)
    LOGGER.addHandler(audit_handler)


def detect_environment() -> dict[str, Any]:
    """Return and print all laptop/GPU properties needed to reproduce a run."""

    import torch
    import transformers

    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_count else None
    total_vram = None
    free_vram = None
    if gpu_count:
        properties = torch.cuda.get_device_properties(0)
        total_vram = float(properties.total_memory) / (1024**3)
        free_bytes, _ = torch.cuda.mem_get_info(0)
        free_vram = float(free_bytes) / (1024**3)
    report = {
        "created_utc": utc_now(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "total_gpu_vram_gib": total_vram,
        "available_gpu_vram_gib_at_start": free_vram,
        "bf16_supported": bool(cuda_available and torch.cuda.is_bf16_supported()),
        "fp16_available": bool(cuda_available),
        "flash_attention_or_sdpa": {
            "flash_sdp": bool(
                cuda_available
                and hasattr(torch.backends.cuda, "flash_sdp_enabled")
                and torch.backends.cuda.flash_sdp_enabled()
            ),
            "mem_efficient_sdp": bool(
                cuda_available
                and hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled")
                and torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "sdpa_available": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
        },
        "cpu_threads": int(torch.get_num_threads()),
        "distributed_initialized": bool(
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def set_seed(seed: int, strict_deterministic: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_processor_path(value: Path | None, project_root: Path) -> Path:
    """Find a processor directory without treating any model checkpoint as one."""

    candidates: list[Path] = []
    if value is not None:
        candidates.append(value.expanduser().resolve())
    candidates.extend(
        [
            project_root / "processor",
            project_root / "data" / "processor",
            project_root / "tiny-overfit" / "processor",
        ]
    )
    for candidate in candidates:
        if (candidate / "preprocessor_config.json").is_file() and (
            (candidate / "tokenizer_config.json").is_file()
            or (candidate / "vocab.json").is_file()
        ):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Validated Bengali processor not found. Supply --processor-path pointing "
        f"to the processor with the 73-token contract. Searched: {searched}"
    )


def load_and_audit_processor(path: Path) -> tuple[Any, dict[str, Any]]:
    """Load the existing processor and fail on any contract mismatch."""

    from transformers import Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(path)
    tokenizer = processor.tokenizer
    vocab = {str(token): int(index) for token, index in tokenizer.get_vocab().items()}
    if len(tokenizer) != EXPECTED_VOCAB_SIZE or len(vocab) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Validated Bengali processor must have {EXPECTED_VOCAB_SIZE} tokens; "
            f"got len(tokenizer)={len(tokenizer)}, vocab={len(vocab)}"
        )
    if int(tokenizer.pad_token_id) != EXPECTED_BLANK_ID:
        raise ValueError(f"Expected blank/pad ID 0, got {tokenizer.pad_token_id}")
    if int(tokenizer.unk_token_id) != EXPECTED_UNKNOWN_ID:
        raise ValueError(f"Expected unknown ID 1, got {tokenizer.unk_token_id}")
    delimiter_id = int(tokenizer.convert_tokens_to_ids(EXPECTED_DELIMITER_TOKEN))
    if delimiter_id != EXPECTED_DELIMITER_ID:
        raise ValueError(f"Expected delimiter ID 2, got {delimiter_id}")
    if getattr(tokenizer, "word_delimiter_token", None) != EXPECTED_DELIMITER_TOKEN:
        raise ValueError(
            "Expected tokenizer.word_delimiter_token to be '|'; "
            f"got {getattr(tokenizer, 'word_delimiter_token', None)!r}"
        )
    feature = processor.feature_extractor
    if int(feature.sampling_rate) != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected processor sampling rate {TARGET_SAMPLE_RATE}, got {feature.sampling_rate}"
        )
    if not bool(getattr(feature, "do_normalize", False)):
        raise ValueError("Validated MMS processor must enable waveform normalization")
    ordered_vocab = sorted(vocab.items(), key=lambda pair: pair[1])
    if [index for _, index in ordered_vocab] != list(range(EXPECTED_VOCAB_SIZE)):
        raise ValueError("Processor vocabulary IDs are not contiguous 0..72")
    vocabulary_hash = sha256_text(
        json.dumps(ordered_vocab, ensure_ascii=False, separators=(",", ":"))
    )
    special_tokens = {
        name: getattr(tokenizer, name, None)
        for name in (
            "pad_token",
            "unk_token",
            "word_delimiter_token",
            "bos_token",
            "eos_token",
        )
    }
    audit = {
        "processor_path": str(path),
        "processor_files": sorted(
            str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()
        ),
        "vocabulary_size": EXPECTED_VOCAB_SIZE,
        "vocabulary_sha256": vocabulary_hash,
        "token_to_id": {token: index for token, index in ordered_vocab},
        "id_to_token": {str(index): token for token, index in ordered_vocab},
        "special_tokens": special_tokens,
        "blank_id": EXPECTED_BLANK_ID,
        "unknown_id": EXPECTED_UNKNOWN_ID,
        "delimiter_token": EXPECTED_DELIMITER_TOKEN,
        "delimiter_id": EXPECTED_DELIMITER_ID,
        "sampling_rate": int(feature.sampling_rate),
        "do_normalize": bool(feature.do_normalize),
        "normalization": {
            "function": "asr_dialect_benchmark.tokenization.simple_tokenizer.normalize_bengali_text",
            "nfc": True,
            "zero_width_removed": True,
            "bengali_codepoints_and_spaces_only": True,
        },
    }
    return processor, audit


def split_archive_path(value: str) -> tuple[Path, str] | None:
    if "::" not in value:
        return None
    archive, member = value.split("::", 1)
    return Path(archive).expanduser(), member


def resolve_audio_value(value: str, manifest_path: Path, project_root: Path) -> str:
    archive_value = split_archive_path(value)
    if archive_value is not None:
        archive, member = archive_value
        if not archive.is_absolute():
            archive = (manifest_path.parent / archive).resolve()
        return f"{archive}::{member}"
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    candidates = [(manifest_path.parent / path).resolve(), (project_root / path).resolve()]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def source_bytes(source: str) -> bytes:
    archive_value = split_archive_path(source)
    if archive_value is None:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        return path.read_bytes()
    archive_path, member = archive_value
    if not archive_path.is_file():
        raise FileNotFoundError(f"Audio archive does not exist: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        try:
            return archive.read(member)
        except KeyError as exc:
            raise FileNotFoundError(f"Audio member does not exist: {archive_path}::{member}") from exc


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(source_rate), int(target_rate))
        return resample_poly(
            audio, target_rate // gcd, source_rate // gcd
        ).astype(np.float32, copy=False)
    except Exception:
        # This fallback is only for environments without scipy; it does not
        # normalize or clip the waveform.
        old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_length = max(1, round(len(audio) * target_rate / source_rate))
        new_x = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
        return np.interp(new_x, old_x, audio).astype(np.float32, copy=False)


def read_audio_source(source: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    archive_value = split_archive_path(source)
    if archive_value is None:
        handle: Any = source
    else:
        handle = io.BytesIO(source_bytes(source))
    audio, sample_rate = sf.read(handle, dtype="float32", always_2d=True)
    if audio.ndim != 2 or audio.shape[1] == 0:
        raise ValueError("decoded audio has no channels")
    audio = audio.mean(axis=1).astype(np.float32, copy=False)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("audio is empty or contains non-finite values")
    audio = resample_audio(audio, int(sample_rate), TARGET_SAMPLE_RATE)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("resampled audio is empty or non-finite")
    if float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) <= 1e-6:
        raise ValueError("audio is effectively silent")
    return audio, TARGET_SAMPLE_RATE


def tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(index) for index in ids]


def decode_target(tokenizer: Any, ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(ids), group_tokens=False, skip_special_tokens=True
    ).strip()


def parse_manifest(path: Path, project_root: Path, manually_verified: bool) -> list[Sample]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32:
        raise ValueError(f"Tiny manifest must contain exactly 32 rows; got {len(rows)}")
    required = {"sample_id", "transcript"}
    if not required.issubset(rows[0] if rows else {}):
        raise ValueError("Manifest must contain sample_id and transcript columns")
    audio_column = "audio_path" if "audio_path" in rows[0] else "audio"
    if audio_column not in rows[0]:
        raise ValueError("Manifest must contain audio_path (or legacy audio) column")
    if not manually_verified:
        raise RuntimeError(
            "Pass --manually-verified only after listening to all 32 pairs and checking transcripts"
        )
    samples: list[Sample] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"Manifest row {row_number} has a missing or duplicate sample_id")
        seen.add(sample_id)
        verified = str(row.get("manually_verified", "NO")).strip().upper()
        if verified != "YES":
            raise ValueError(f"{sample_id}: manually_verified must be YES, got {verified!r}")
        raw = str(row.get("transcript", ""))
        normalized = normalize_bengali_text(raw)
        if not normalized:
            raise ValueError(f"{sample_id}: transcript is empty after normalization")
        audio = resolve_audio_value(str(row.get(audio_column, "")).strip(), path, project_root)
        district = str(row.get("district", "")).strip()
        dialect = str(row.get("dialect", "")).strip()
        expected = DISTRICT_TO_DIALECT.get(district, "")
        if district and expected and dialect and dialect.casefold() != expected.casefold():
            raise ValueError(
                f"{sample_id}: dialect {dialect!r} disagrees with district {district!r} ({expected})"
            )
        duration = row.get("duration_seconds") or row.get("duration") or ""
        samples.append(
            Sample(
                sample_id=sample_id,
                audio_path=audio,
                transcript=normalized,
                dialect=dialect or expected,
                district=district,
                manually_verified=verified,
                duration_seconds=float(duration) if duration else None,
                raw_transcript=raw,
                normalized_transcript=normalized,
                raw_changed=raw != normalized,
                resolved_audio_path=audio,
            )
        )
    return samples


def audit_samples(
    samples: list[Sample],
    processor: Any,
    processor_audit: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[Sample], dict[str, Any]]:
    tokenizer = processor.tokenizer
    target_rows: list[dict[str, Any]] = []
    duration_values: list[float] = []
    character_values: list[int] = []
    word_values: list[int] = []
    for sample in samples:
        try:
            raw_bytes = source_bytes(sample.audio_path)
            audio, sample_rate = read_audio_source(sample.audio_path)
            duration = len(audio) / sample_rate
            if args.max_audio_seconds is not None and duration > args.max_audio_seconds:
                raise ValueError(
                    f"duration {duration:.3f}s exceeds --max-audio-seconds {args.max_audio_seconds:.3f}; "
                    "the script never crops an audio/transcript pair"
                )
            if sample.duration_seconds is not None and abs(sample.duration_seconds - duration) > 0.25:
                raise ValueError(
                    f"manifest duration {sample.duration_seconds:.3f}s disagrees with decoded {duration:.3f}s"
                )
            ids = tokenizer_ids(tokenizer, sample.transcript)
            if not ids:
                raise ValueError("encoded target is empty")
            if any(index in {EXPECTED_BLANK_ID, EXPECTED_UNKNOWN_ID, -100} for index in ids):
                raise ValueError(f"encoded target contains blank, unknown, or padding ID: {ids}")
            decoded = decode_target(tokenizer, ids)
            if normalize_bengali_text(decoded) != sample.transcript:
                raise ValueError(
                    f"target round-trip mismatch: normalized={sample.transcript!r}, decoded={decoded!r}"
                )
            repeat_count = int(sum(left == right for left, right in zip(ids, ids[1:])))
            sample.duration_seconds = float(duration)
            sample.audio_sha256 = sha256_bytes(raw_bytes)
            sample.transcript_sha256 = sha256_text(sample.transcript)
            sample.target_ids = tuple(ids)
            sample.decoded_target = decoded
            duration_values.append(duration)
            character_values.append(len(sample.transcript.replace(" ", "")))
            word_values.append(len(sample.transcript.split()))
            target_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "audio_path": sample.audio_path,
                    "raw_transcript": sample.raw_transcript,
                    "normalized_transcript": sample.transcript,
                    "encoded_token_ids": json.dumps(ids),
                    "decoded_target": decoded,
                    "raw_changed": sample.raw_changed,
                    "duration_seconds": duration,
                    "audio_sha256": sample.audio_sha256,
                    "transcript_sha256": sample.transcript_sha256,
                    "adjacent_repeat_count": repeat_count,
                }
            )
        except Exception as exc:
            raise ValueError(f"{sample.sample_id} ({sample.audio_path}): {exc}") from exc
    target_audit = output_dir / "tiny_target_audit.csv"
    with target_audit.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)
    locked = output_dir / "tiny_32_manifest_locked.csv"
    locked_fields = [
        "sample_id",
        "audio_path",
        "transcript",
        "dialect",
        "district",
        "duration_seconds",
        "audio_sha256",
        "transcript_sha256",
        "manually_verified",
    ]
    with locked.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=locked_fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: getattr(sample, field) for field in locked_fields})
    manifest_hash = sha256_bytes(locked.read_bytes())
    metadata = {
        "created_utc": utc_now(),
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "manifest_sha256": manifest_hash,
        "audio_sha256": {sample.sample_id: sample.audio_sha256 for sample in samples},
        "transcript_sha256": {sample.sample_id: sample.transcript_sha256 for sample in samples},
        "processor_vocabulary_sha256": processor_audit["vocabulary_sha256"],
        "seed": args.seed,
        "project_commit": git_commit(args.project_root),
        "manual_verification_required": True,
    }
    atomic_json(output_dir / "tiny_32_manifest_metadata.json", metadata)
    audit = {
        "sample_count": len(samples),
        "total_duration_seconds": float(sum(duration_values)),
        "min_duration_seconds": float(min(duration_values)),
        "max_duration_seconds": float(max(duration_values)),
        "mean_duration_seconds": float(np.mean(duration_values)),
        "transcript_characters_total": int(sum(character_values)),
        "transcript_characters_min": int(min(character_values)),
        "transcript_characters_max": int(max(character_values)),
        "transcript_words_total": int(sum(word_values)),
        "transcript_words_min": int(min(word_values)),
        "transcript_words_max": int(max(word_values)),
        "manifest_sha256": manifest_hash,
        "rejected_samples": [],
    }
    atomic_json(output_dir / "tiny_data_audit.json", audit)
    return samples, audit


def auto_defaults(environment: dict[str, Any], args: argparse.Namespace) -> None:
    vram = environment.get("total_gpu_vram_gib") or 0.0
    if args.batch_size is None:
        args.batch_size = 1 if vram < 8 else 2
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 8 if vram < 8 else 4
    if args.trainable_encoder_layers is None:
        args.trainable_encoder_layers = 2 if vram < 6 else 4
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = bool(vram < 12)
    if args.precision is None:
        args.precision = "fp16" if environment["cuda_available"] else "none"
    if args.num_workers is None:
        args.num_workers = 0 if os.name == "nt" else min(2, os.cpu_count() or 1)
    if args.precision == "bf16" and not environment["bf16_supported"]:
        raise RuntimeError("--bf16 requested but this GPU does not support BF16")
    if args.precision == "fp16" and not environment["fp16_available"]:
        raise RuntimeError("--fp16 requested but CUDA/FP16 is unavailable")
    if not environment["cuda_available"]:
        args.precision = "none"


def build_model(args: argparse.Namespace, device: Any) -> tuple[Any, dict[str, Any]]:
    """Load only the public MMS backbone and initialize a fresh 73-way head."""

    import torch
    from transformers import Wav2Vec2ForCTC

    if args.model_name != "facebook/mms-300m":
        LOGGER.warning("Model source override: %s", args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        ignore_mismatched_sizes=True,
        vocab_size=EXPECTED_VOCAB_SIZE,
        pad_token_id=EXPECTED_BLANK_ID,
    )
    hidden_size = int(model.config.hidden_size)
    # Never reuse a downloaded CTC head.  Only the MMS acoustic backbone is
    # loaded; this linear layer is freshly initialized for the validated vocab.
    model.lm_head = torch.nn.Linear(hidden_size, EXPECTED_VOCAB_SIZE)
    model.config.vocab_size = EXPECTED_VOCAB_SIZE
    model.config.pad_token_id = EXPECTED_BLANK_ID
    model.config.ctc_loss_reduction = "mean"
    model.config.ctc_zero_infinity = False
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True
    wav2vec = model.wav2vec2
    feature_extractor = getattr(wav2vec, "feature_extractor", None)
    if feature_extractor is None:
        raise RuntimeError("MMS backbone does not expose a convolutional feature encoder")
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False
    layers = getattr(getattr(wav2vec, "encoder", None), "layers", None)
    if layers is None:
        raise RuntimeError("MMS backbone does not expose transformer encoder layers")
    layer_count = len(layers)
    requested = int(args.trainable_encoder_layers)
    if requested < 1 or requested > layer_count:
        raise ValueError(f"--trainable-encoder-layers must be 1..{layer_count}, got {requested}")
    for layer in layers[-requested:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    if any("moe" in name.lower() or "router" in name.lower() or "expert" in name.lower() for name, _ in model.named_modules()):
        raise AssertionError("Plain MMS model unexpectedly contains an MoE/router/expert module")
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not any(name.startswith("lm_head.") for name in trainable_names):
        raise AssertionError("Fresh CTC head is not trainable")
    if not any("encoder.layers" in name for name in trainable_names):
        raise AssertionError("No transformer encoder layer is trainable")
    feature_count = sum(p.numel() for p in feature_extractor.parameters())
    encoder_count = sum(p.numel() for p in wav2vec.encoder.parameters())
    head_count = sum(p.numel() for p in model.lm_head.parameters())
    total_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    audit = {
        "model_source": args.model_name,
        "output_vocabulary_size": EXPECTED_VOCAB_SIZE,
        "blank_id": EXPECTED_BLANK_ID,
        "unknown_id": EXPECTED_UNKNOWN_ID,
        "delimiter_id": EXPECTED_DELIMITER_ID,
        "total_parameters": int(total_count),
        "trainable_parameters": int(trainable_count),
        "frozen_parameters": int(total_count - trainable_count),
        "feature_encoder_parameters": int(feature_count),
        "transformer_encoder_parameters": int(encoder_count),
        "ctc_head_parameters": int(head_count),
        "transformer_layer_count": layer_count,
        "trainable_transformer_layers": list(range(layer_count - requested, layer_count)),
        "trainable_parameter_names": trainable_names,
        "moe_components_present": False,
        "dialect_classifier_present": False,
        "augmentation_present": False,
        "used_failed_checkpoint": False,
    }
    atomic_json(args.output_dir / "model_audit.json", audit)
    LOGGER.info(
        "Model: %s total=%d trainable=%d frozen=%d; trainable layers=%s",
        args.model_name,
        total_count,
        trainable_count,
        total_count - trainable_count,
        audit["trainable_transformer_layers"],
    )
    return model, audit


def feature_output_lengths(model: Any, input_lengths: Any) -> Any:
    encoder = getattr(model, "wav2vec2", model)
    method = getattr(encoder, "_get_feat_extract_output_lengths", None)
    if method is None:
        method = getattr(model, "_get_feat_extract_output_lengths", None)
    if method is None:
        raise RuntimeError("MMS model does not provide official feature output lengths")
    return method(input_lengths).long()


def make_batch(
    indices: Sequence[int],
    samples: list[Sample],
    processor: Any,
    device: Any,
) -> dict[str, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    arrays = []
    for index in indices:
        audio, rate = read_audio_source(samples[index].audio_path)
        if rate != TARGET_SAMPLE_RATE:
            raise AssertionError("audio loader failed to return 16 kHz")
        arrays.append(audio)
    processed = processor(
        arrays,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    if "attention_mask" not in processed:
        raise RuntimeError("validated MMS processor did not return an attention mask")
    labels = [torch.tensor(samples[index].target_ids, dtype=torch.long) for index in indices]
    padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    target_lengths = torch.tensor([len(sample) for sample in labels], dtype=torch.long)
    return {
        "input_values": processed["input_values"].to(device),
        "attention_mask": processed["attention_mask"].to(device),
        "input_lengths": processed["attention_mask"].sum(-1).long().to(device),
        "targets": padded.to(device),
        "target_lengths": target_lengths.to(device),
        "sample_indices": list(indices),
    }


def validate_ctc_lengths(model: Any, batch: dict[str, Any], output_time: int) -> tuple[Any, Any]:
    import torch

    output_lengths = feature_output_lengths(model, batch["input_lengths"])
    if int(output_lengths.max().item()) > output_time:
        raise RuntimeError(
            f"Official CTC lengths exceed logits time dimension: {output_lengths.tolist()} > {output_time}"
        )
    flat_targets: list[Any] = []
    required: list[int] = []
    offset = 0
    for row, target_length in enumerate(batch["target_lengths"].tolist()):
        target = batch["targets"][row, :target_length]
        if (target == EXPECTED_BLANK_ID).any() or (target == EXPECTED_UNKNOWN_ID).any() or (target == -100).any():
            raise RuntimeError(f"Invalid target IDs for batch sample index {batch['sample_indices'][row]}")
        repeats = int((target[1:] == target[:-1]).sum().item())
        required.append(int(target_length) + repeats)
        flat_targets.append(target)
        offset += int(target_length)
    flat = torch.cat(flat_targets).long() if flat_targets else torch.empty(0, dtype=torch.long, device=batch["targets"].device)
    required_tensor = torch.tensor(required, dtype=torch.long, device=output_lengths.device)
    if (output_lengths < required_tensor).any():
        details = [
            {
                "sample_index": batch["sample_indices"][index],
                "input_length": int(output_lengths[index].item()),
                "required_length": int(required_tensor[index].item()),
                "target_length": int(batch["target_lengths"][index].item()),
            }
            for index in range(len(required))
            if output_lengths[index] < required_tensor[index]
        ]
        raise RuntimeError(f"CTC repeated-label length constraint failed: {details}")
    return flat, output_lengths


def ctc_forward(model: Any, batch: dict[str, Any], autocast_context: Any) -> tuple[Any, Any, Any, Any]:
    import torch
    import torch.nn.functional as F

    with autocast_context:
        outputs = model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
    logits_float = logits.float()
    if logits_float.ndim != 3 or logits_float.shape[-1] != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(f"Expected logits [B,T,73], got {tuple(logits_float.shape)}")
    flat_targets, output_lengths = validate_ctc_lengths(model, batch, logits_float.shape[1])
    loss = F.ctc_loss(
        logits_float.log_softmax(-1).transpose(0, 1),
        flat_targets,
        output_lengths,
        batch["target_lengths"],
        blank=EXPECTED_BLANK_ID,
        zero_infinity=False,
        reduction="mean",
    )
    if not torch.isfinite(loss).item() or not torch.isfinite(logits_float).all().item():
        raise FloatingPointError("CTC loss or logits became non-finite")
    return loss, logits_float, output_lengths, flat_targets


def collapse_ctc(ids: Iterable[int]) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None
    for raw in ids:
        token_id = int(raw)
        if token_id == previous:
            continue
        previous = token_id
        if token_id != EXPECTED_BLANK_ID:
            collapsed.append(token_id)
    return collapsed


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, ref in enumerate(reference, start=1):
        current = [index]
        for other, hyp in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[other] + 1,
                    previous[other - 1] + (ref != hyp),
                )
            )
        previous = current
    return previous[-1]


def grad_norm(parameters: Iterable[Any]) -> float:
    import torch

    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        values = parameter.grad.detach().float()
        if not torch.isfinite(values).all():
            return float("nan")
        total += float(values.pow(2).sum().item())
    return math.sqrt(total)


def parameter_changed(before: dict[str, Any], model: Any) -> bool:
    for name, old in before.items():
        current = dict(model.named_parameters())[name].detach()
        if not bool((current != old).any().item()):
            continue
        return True
    return False


def prediction_metrics(
    processor: Any,
    samples: list[Sample],
    predictions: list[dict[str, Any]],
    frame_count: int,
    counts: dict[int, int],
    blank_probability_sum: float,
    delimiter_probability_sum: float,
    entropy_sum: float,
    loss: float,
) -> dict[str, Any]:
    references = [sample.transcript for sample in samples]
    character_denominator = sum(len(ref.replace(" ", "")) for ref in references)
    word_denominator = sum(len(ref.split()) for ref in references)
    character_errors = sum(
        edit_distance(list(ref.replace(" ", "")), list(row["prediction"].replace(" ", "")))
        for ref, row in zip(references, predictions)
    )
    word_errors = sum(
        edit_distance(ref.split(), row["prediction"].split())
        for ref, row in zip(references, predictions)
    )
    blank_fraction = counts.get(EXPECTED_BLANK_ID, 0) / max(1, frame_count)
    delimiter_fraction = counts.get(EXPECTED_DELIMITER_ID, 0) / max(1, frame_count)
    return {
        "eval_ctc_loss": float(loss),
        "cer": character_errors / max(1, character_denominator),
        "wer": word_errors / max(1, word_denominator),
        "empty_prediction_rate": sum(not row["prediction"] for row in predictions) / max(1, len(predictions)),
        "mean_prediction_length_chars": float(np.mean([len(row["prediction"].replace(" ", "")) for row in predictions])),
        "mean_prediction_length_words": float(np.mean([len(row["prediction"].split()) for row in predictions])),
        "blank_argmax_fraction": blank_fraction,
        "delimiter_argmax_fraction": delimiter_fraction,
        "unknown_argmax_fraction": counts.get(EXPECTED_UNKNOWN_ID, 0) / max(1, frame_count),
        "mean_blank_probability": blank_probability_sum / max(1, frame_count),
        "mean_delimiter_probability": delimiter_probability_sum / max(1, frame_count),
        "frame_entropy": entropy_sum / max(1, frame_count),
        "valid_frame_count": int(frame_count),
        "raw_argmax_distribution": {str(key): int(value) for key, value in sorted(counts.items())},
        "predictions": predictions,
    }


def evaluate(
    model: Any,
    samples: list[Sample],
    processor: Any,
    device: Any,
    batch_size: int,
    precision: str,
) -> dict[str, Any]:
    import torch

    model.eval()
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    use_autocast = device.type == "cuda" and precision in {"fp16", "bf16"}
    predictions: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    blank_probability_sum = 0.0
    delimiter_probability_sum = 0.0
    entropy_sum = 0.0
    frame_count = 0
    loss_sum = 0.0
    trace_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            indices = list(range(start, min(len(samples), start + batch_size)))
            batch = make_batch(indices, samples, processor, device)
            context = torch.autocast("cuda", dtype=autocast_dtype) if use_autocast else nullcontext()
            loss, logits, output_lengths, _ = ctc_forward(model, batch, context)
            loss_sum += float(loss.item()) * len(indices)
            probabilities = logits.softmax(-1)
            frame_ids = logits.argmax(-1)
            for local, sample_index in enumerate(indices):
                length = int(output_lengths[local].item())
                raw_ids = frame_ids[local, :length].detach().cpu().tolist()
                raw_probs = probabilities[local, :length]
                for token_id in raw_ids:
                    counts[token_id] = counts.get(token_id, 0) + 1
                frame_count += length
                blank_probability_sum += float(raw_probs[:, EXPECTED_BLANK_ID].sum().item())
                delimiter_probability_sum += float(raw_probs[:, EXPECTED_DELIMITER_ID].sum().item())
                entropy_sum += float((-(raw_probs * raw_probs.clamp_min(1e-12).log()).sum(-1)).sum().item())
                collapsed = collapse_ctc(raw_ids)
                prediction = processor.tokenizer.decode(
                    collapsed, group_tokens=False, skip_special_tokens=True
                ).strip()
                sample = samples[sample_index]
                reference = sample.transcript
                row = {
                    "step": None,
                    "sample_id": sample.sample_id,
                    "audio_path": sample.audio_path,
                    "dialect": sample.dialect,
                    "district": sample.district,
                    "reference": reference,
                    "prediction": prediction,
                    "reference_length_chars": len(reference.replace(" ", "")),
                    "prediction_length_chars": len(prediction.replace(" ", "")),
                    "reference_length_words": len(reference.split()),
                    "prediction_length_words": len(prediction.split()),
                    "cer": edit_distance(list(reference.replace(" ", "")), list(prediction.replace(" ", ""))) / max(1, len(reference.replace(" ", ""))),
                    "wer": edit_distance(reference.split(), prediction.split()) / max(1, len(reference.split())),
                    "empty_prediction": not prediction,
                    "raw_argmax_token_count": len(raw_ids),
                    "raw_blank_fraction": raw_ids.count(EXPECTED_BLANK_ID) / max(1, len(raw_ids)),
                    "raw_delimiter_fraction": raw_ids.count(EXPECTED_DELIMITER_ID) / max(1, len(raw_ids)),
                    "mean_blank_probability": float(raw_probs[:, EXPECTED_BLANK_ID].mean().item()),
                    "mean_delimiter_probability": float(raw_probs[:, EXPECTED_DELIMITER_ID].mean().item()),
                }
                predictions.append(row)
                if len(trace_rows) < 4:
                    trace_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "raw_argmax_ids": raw_ids,
                            "collapsed_ids": collapsed,
                        }
                    )
    model.train()
    metrics = prediction_metrics(
        processor,
        samples,
        predictions,
        frame_count,
        counts,
        blank_probability_sum,
        delimiter_probability_sum,
        entropy_sum,
        loss_sum / len(samples),
    )
    metrics["predictions"] = predictions
    metrics["raw_traces"] = trace_rows
    return metrics


def write_predictions(output_dir: Path, step: int, predictions: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> None:
    prediction_dir = output_dir / "predictions"
    trace_dir = output_dir / "token_traces"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    fields = list(predictions[0]) if predictions else []
    snapshot = prediction_dir / f"ctc_predictions_step_{step:06d}.csv"
    with snapshot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    shutil.copyfile(snapshot, output_dir / "ctc_predictions_latest.csv")
    atomic_json(trace_dir / f"raw_tokens_step_{step:06d}.json", trace_rows)


def optimizer_setup(model: Any, args: argparse.Namespace) -> tuple[Any, Any, list[Any], list[Any]]:
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    head_decay, head_no_decay, encoder_decay, encoder_no_decay = [], [], [], []
    head_parameters, encoder_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_no_decay = name.endswith("bias") or "layer_norm" in name.lower() or "layernorm" in name.lower()
        if name.startswith("lm_head."):
            head_parameters.append(parameter)
            (head_no_decay if is_no_decay else head_decay).append(parameter)
        else:
            encoder_parameters.append(parameter)
            (encoder_no_decay if is_no_decay else encoder_decay).append(parameter)
    groups = [
        {"params": head_decay, "lr": args.head_lr, "weight_decay": args.weight_decay, "group_name": "ctc_head_decay"},
        {"params": head_no_decay, "lr": args.head_lr, "weight_decay": 0.0, "group_name": "ctc_head_no_decay"},
        {"params": encoder_decay, "lr": args.encoder_lr, "weight_decay": args.weight_decay, "group_name": "encoder_decay"},
        {"params": encoder_no_decay, "lr": args.encoder_lr, "weight_decay": 0.0, "group_name": "encoder_no_decay"},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    warmup = max(0, int(args.warmup_steps))

    def schedule(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(1.0, float(step + 1) / warmup)

    scheduler = LambdaLR(optimizer, schedule)
    return optimizer, scheduler, head_parameters, encoder_parameters


def save_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    step: int,
    best_metrics: dict[str, Any] | None,
    args: argparse.Namespace,
    processor_audit: dict[str, Any],
    manifest_hash: str,
    environment: dict[str, Any],
) -> None:
    import torch

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "global_optimizer_step": int(step),
        "epoch": int(step),
        "best_metrics": best_metrics,
        "processor_path": processor_audit["processor_path"],
        "processor_vocabulary_sha256": processor_audit["vocabulary_sha256"],
        "manifest_sha256": manifest_hash,
        "training_arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "random_seed": args.seed,
        "environment": environment,
        "model_source": args.model_name,
        "used_failed_checkpoint": False,
        "moe_components_present": False,
        "dialect_loss_present": False,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def gate_passed(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["cer"] <= GATE_CER
        and metrics["wer"] <= GATE_WER
        and metrics["empty_prediction_rate"] < GATE_EMPTY_RATE
    )


def status_payload(
    status: str,
    args: argparse.Namespace,
    latest_step: int,
    latest_metrics: dict[str, Any],
    best_metrics: dict[str, Any] | None,
    gate_metrics: dict[str, Any] | None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "passed": bool(gate_metrics),
        "reason": reason,
        "best_step": best_metrics.get("step") if best_metrics else None,
        "best_checkpoint": str(args.output_dir / "tiny_overfit_best.pt") if best_metrics else None,
        "best_cer": best_metrics.get("cer") if best_metrics else None,
        "best_wer": best_metrics.get("wer") if best_metrics else None,
        "best_empty_prediction_rate": best_metrics.get("empty_prediction_rate") if best_metrics else None,
        "best_same_checkpoint_gate_metrics": {
            "step": gate_metrics.get("step") if gate_metrics else None,
            "cer": gate_metrics.get("cer") if gate_metrics else None,
            "wer": gate_metrics.get("wer") if gate_metrics else None,
            "empty_prediction_rate": gate_metrics.get("empty_prediction_rate") if gate_metrics else None,
        },
        "latest_step": int(latest_step),
        "latest_metrics": latest_metrics,
        "processor_vocabulary_size": EXPECTED_VOCAB_SIZE,
        "blank_id": EXPECTED_BLANK_ID,
        "unknown_id": EXPECTED_UNKNOWN_ID,
        "delimiter_id": EXPECTED_DELIMITER_ID,
        "model_source": args.model_name,
        "used_moe_checkpoint": False,
        "moe_components_present": False,
        "dialect_loss_present": False,
        "augmentation_present": False,
    }


def json_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def run_experiment(args: argparse.Namespace) -> int:
    import torch

    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()
    setup_logging(args.output_dir)
    environment = detect_environment()
    if environment["distributed_initialized"]:
        raise RuntimeError("Distributed training is already initialized; plain test refuses to continue")
    if not environment["cuda_available"] and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Use a laptop GPU or pass --allow-cpu explicitly.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto_defaults(environment, args)
    if environment["gpu_count"] > 1:
        LOGGER.warning("%d GPUs are visible; using only device 0 and no DDP/DataParallel", environment["gpu_count"])
    set_seed(args.seed, args.strict_deterministic)
    processor_path = resolve_processor_path(args.processor_path, args.project_root)
    processor, processor_audit = load_and_audit_processor(processor_path)
    args.processor_path = processor_path
    atomic_json(args.output_dir / "environment.json", {**environment, "selected_device": str(device), "precision": args.precision})
    atomic_json(args.output_dir / "processor_audit.json", processor_audit)
    atomic_json(args.output_dir / "run_config.json", {"created_utc": utc_now(), **json_args(args), "environment": environment})
    samples = parse_manifest(args.manifest.resolve(), args.project_root, args.manually_verified)
    samples, data_audit = audit_samples(samples, processor, processor_audit, args, args.output_dir)
    model, model_audit = build_model(args, device)
    optimizer, scheduler, head_parameters, encoder_parameters = optimizer_setup(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "fp16" and device.type == "cuda")
    manifest_hash = sha256_bytes((args.output_dir / "tiny_32_manifest_locked.csv").read_bytes())
    history_path = args.output_dir / "tiny_overfit_history.jsonl"
    status_path = args.output_dir / "tiny_overfit_status.json"
    best_metrics: dict[str, Any] | None = None
    gate_metrics: dict[str, Any] | None = None
    latest_metrics: dict[str, Any] = {}
    best_cer = float("inf")

    def append_record(record: dict[str, Any]) -> None:
        nonlocal best_metrics, gate_metrics, best_cer, latest_metrics
        predictions = record.pop("predictions", [])
        traces = record.pop("raw_traces", [])
        for prediction in predictions:
            prediction["step"] = int(record["step"])
        latest_metrics = dict(record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_predictions(args.output_dir, int(record["step"]), predictions, traces)
        if record["cer"] < best_cer:
            best_cer = record["cer"]
            best_metrics = dict(record)
            shutil.copyfile(
                args.output_dir / "predictions" / f"ctc_predictions_step_{int(record['step']):06d}.csv",
                args.output_dir / "ctc_predictions_best.csv",
            )
        if gate_passed(record) and (gate_metrics is None or record["cer"] < gate_metrics["cer"]):
            gate_metrics = dict(record)
        atomic_json(status_path, status_payload("running", args, int(record["step"]), record, best_metrics, gate_metrics))
        LOGGER.info(
            "step=%d loss=%.5f CER=%.4f WER=%.4f empty=%.3f blank=%.4f delimiter=%.4f head_grad=%s encoder_grad=%s",
            record["step"], record["train_loss"] if record.get("train_loss") is not None else float("nan"), record["cer"], record["wer"],
            record["empty_prediction_rate"], record["blank_argmax_fraction"], record["delimiter_argmax_fraction"],
            record.get("head_grad_norm"), record.get("encoder_grad_norm"),
        )

    def evaluate_and_record(step: int, train_loss: float, head_grad: float | None, encoder_grad: float | None, step_seconds: float = 0.0) -> None:
        eval_start = time.perf_counter()
        metrics = evaluate(model, samples, processor, device, args.batch_size, args.precision)
        record = {
            "step": int(step),
            "train_loss": float(train_loss),
            "head_grad_norm": head_grad,
            "encoder_grad_norm": encoder_grad,
            "total_grad_norm": None if head_grad is None or encoder_grad is None else float(math.sqrt(head_grad**2 + encoder_grad**2)),
            "head_lr": float(max(group["lr"] for group in optimizer.param_groups if group.get("group_name", "").startswith("ctc_head"))),
            "encoder_lr": float(max(group["lr"] for group in optimizer.param_groups if group.get("group_name", "").startswith("encoder"))),
            "gpu_allocated_gib": float(torch.cuda.memory_allocated() / 1024**3) if device.type == "cuda" else 0.0,
            "gpu_reserved_gib": float(torch.cuda.memory_reserved() / 1024**3) if device.type == "cuda" else 0.0,
            "gpu_max_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0,
            "step_seconds": float(step_seconds),
            "evaluation_seconds": float(time.perf_counter() - eval_start),
            **metrics,
        }
        append_record(record)
        atomic_json(args.output_dir / "ctc_collapse_summary.json", {key: value for key, value in record.items() if key not in {"predictions", "raw_traces"}})

    initial = evaluate(model, samples, processor, device, args.batch_size, args.precision)
    initial_record = {
        "step": 0,
        "train_loss": None,
        "head_grad_norm": None,
        "encoder_grad_norm": None,
        "total_grad_norm": None,
        "head_lr": float(max(group["lr"] for group in optimizer.param_groups if group.get("group_name", "").startswith("ctc_head"))),
        "encoder_lr": float(max(group["lr"] for group in optimizer.param_groups if group.get("group_name", "").startswith("encoder"))),
        "gpu_allocated_gib": float(torch.cuda.memory_allocated() / 1024**3) if device.type == "cuda" else 0.0,
        "gpu_reserved_gib": float(torch.cuda.memory_reserved() / 1024**3) if device.type == "cuda" else 0.0,
        "gpu_max_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0,
        "step_seconds": 0.0,
        "evaluation_seconds": 0.0,
        **initial,
    }
    append_record(initial_record)
    save_checkpoint(args.output_dir / "tiny_overfit_last.pt", model, optimizer, scheduler, scaler, 0, best_metrics, args, processor_audit, manifest_hash, environment)
    save_checkpoint(args.output_dir / "tiny_overfit_best.pt", model, optimizer, scheduler, scaler, 0, best_metrics, args, processor_audit, manifest_hash, environment)
    if args.dry_run:
        atomic_json(status_path, status_payload("dry_run", args, 0, latest_metrics, best_metrics, gate_metrics, "No optimizer update requested"))
        return 0

    requested_steps = 3 if args.smoke_test else args.max_steps
    model.train()
    try:
        for step in range(1, requested_steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            before = {
                "lm_head.weight": model.lm_head.weight.detach().clone(),
                f"wav2vec2.encoder.layers.{len(model.wav2vec2.encoder.layers)-1}.attention.k_proj.weight": dict(model.named_parameters())[f"wav2vec2.encoder.layers.{len(model.wav2vec2.encoder.layers)-1}.attention.k_proj.weight"].detach().clone(),
            }
            train_loss_sum = 0.0
            head_norm = None
            encoder_norm = None
            for micro in range(args.gradient_accumulation_steps):
                start = ((step - 1) * args.gradient_accumulation_steps + micro) * args.batch_size
                indices = [(start + offset) % len(samples) for offset in range(args.batch_size)]
                batch = make_batch(indices, samples, processor, device)
                context = torch.autocast("cuda", dtype=torch.float16 if args.precision == "fp16" else torch.bfloat16) if device.type == "cuda" and args.precision in {"fp16", "bf16"} else nullcontext()
                loss, logits, _, _ = ctc_forward(model, batch, context)
                train_loss_sum += float(loss.item())
                scaled_loss = loss / args.gradient_accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            head_norm = grad_norm(head_parameters)
            encoder_norm = grad_norm(encoder_parameters)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(torch.as_tensor(total_norm)).item():
                raise FloatingPointError("gradient norm became non-finite")
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            if not parameter_changed(before, model):
                raise RuntimeError("No selected model parameter changed after optimizer update")
            step_seconds = time.perf_counter() - step_start
            if step == 1 or step % args.eval_steps == 0 or step == requested_steps:
                evaluate_and_record(step, train_loss_sum / args.gradient_accumulation_steps, head_norm, encoder_norm, step_seconds)
            save_checkpoint(args.output_dir / "tiny_overfit_last.pt", model, optimizer, scheduler, scaler, step, best_metrics, args, processor_audit, manifest_hash, environment)
            if best_metrics and int(best_metrics["step"]) == step:
                save_checkpoint(args.output_dir / "tiny_overfit_best.pt", model, optimizer, scheduler, scaler, step, best_metrics, args, processor_audit, manifest_hash, environment)
    except KeyboardInterrupt:
        atomic_json(status_path, status_payload("interrupted", args, int(latest_metrics.get("step", 0)), latest_metrics, best_metrics, gate_metrics, "User interrupted the run"))
        raise
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        error_path = args.output_dir / "logs" / "error_traceback.txt"
        atomic_write_text(error_path, traceback.format_exc())
        atomic_json(status_path, status_payload("failed", args, int(latest_metrics.get("step", 0)), latest_metrics, best_metrics, gate_metrics, "CUDA out of memory; reduce batch size/layers or enable checkpointing"))
        raise RuntimeError("CUDA out of memory. See logs/error_traceback.txt and reduce memory settings.") from exc
    except Exception:
        atomic_write_text(args.output_dir / "logs" / "error_traceback.txt", traceback.format_exc())
        atomic_json(status_path, status_payload("failed", args, int(latest_metrics.get("step", 0)), latest_metrics, best_metrics, gate_metrics, "Run raised an exception; see logs/error_traceback.txt"))
        raise
    final_status = "passed" if gate_metrics else "failed"
    reason = None if gate_metrics else "No single checkpoint met CER <= 0.05, WER <= 0.05, and empty rate < 0.10"
    atomic_json(status_path, status_payload(final_status, args, int(latest_metrics.get("step", requested_steps)), latest_metrics, best_metrics, gate_metrics, reason))
    LOGGER.info("GATE 1 %s", "PASSED" if gate_metrics else "FAILED")
    return 0 if gate_metrics else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Exactly 32 manually verified rows")
    parser.add_argument("--processor-path", type=Path, help="Validated Bengali 73-token MMS processor directory")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/mms-300m")
    parser.add_argument("--trainable-encoder-layers", type=int, default=None)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", dest="precision", action="store_const", const="fp16", default=None)
    parser.add_argument("--bf16", dest="precision", action="store_const", const="bf16")
    parser.add_argument("--no-mixed-precision", dest="precision", action="store_const", const="none")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--manually-verified", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_experiment(args)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
