#!/usr/bin/env python
"""Fresh MMS-CTC-compatible three-phase Bengali dialect MoE trainer."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import shutil
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup,
)

TARGET_SAMPLE_RATE = 16_000
EXPECTED_CTC_BLANK_ID = 0
EXPECTED_CTC_VOCAB_SIZE = 73
PREPROCESSING_VERSION = "mms-ctc-bengali-delimiter-v3-contract-audit"
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
ANNOTATION_RE = re.compile(r"<[^>]*>")
WHITESPACE_RE = re.compile(r"\s+")

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
DIALECT_TO_ID = {"Kamrupi": 0, "Jharkhandi": 1, "Varendri": 2, "Rarhi": 3}


def normalize_bengali_text(text: object) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    text = ZERO_WIDTH_RE.sub("", text)
    text = ANNOTATION_RE.sub(" ", text)
    cleaned = []
    for character in text:
        if character.isspace():
            cleaned.append(" ")
        elif (
            "\u0980" <= character <= "\u09FF"
            and unicodedata.category(character)[0] in {"L", "M", "N"}
        ):
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
        f"{root} must contain train/validation/test directories or "
        "train.zip/validation.zip/test.zip"
    )


def iter_text_records(root: Path, split: str) -> Iterator[tuple[str, str, str]]:
    """Yield district, sample ID and raw transcript without opening audio."""
    mode = split_layout(root)
    if mode == "directories":
        split_dir = root / split
        for district in DISTRICT_TO_DIALECT:
            district_dir = split_dir / district
            if not district_dir.is_dir():
                raise RuntimeError(f"Missing required district: {district_dir}")
            for txt_path in sorted(district_dir.rglob("*.txt")):
                relative = txt_path.relative_to(district_dir).with_suffix("").as_posix()
                if not txt_path.with_suffix(".wav").is_file():
                    raise RuntimeError(f"Transcript has no matching WAV: {txt_path}")
                yield (
                    district,
                    f"{district}/{relative}",
                    txt_path.read_text(encoding="utf-8-sig", errors="strict"),
                )
        return

    with zipfile.ZipFile(root / f"{split}.zip") as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        for member in sorted(names):
            parts = list(PurePosixPath(member).parts)
            if parts and parts[0].lower() == split.lower():
                parts = parts[1:]
            if len(parts) < 2 or parts[0] not in DISTRICT_TO_DIALECT:
                continue
            relative = PurePosixPath(*parts[1:])
            if relative.suffix.lower() != ".txt":
                continue
            wav_relative = relative.with_suffix(".wav")
            candidates = {
                str(PurePosixPath(split, parts[0], wav_relative)),
                str(PurePosixPath(parts[0], wav_relative)),
            }
            if not candidates.intersection(names):
                raise RuntimeError(f"ZIP transcript has no matching WAV: {member}")
            yield (
                parts[0],
                f"{parts[0]}/{relative.with_suffix('').as_posix()}",
                archive.read(member).decode("utf-8-sig", errors="strict"),
            )


def validate_processor(processor: Wav2Vec2Processor) -> None:
    tokenizer = processor.tokenizer
    vocabulary = tokenizer.get_vocab()
    if " " in vocabulary:
        raise ValueError("Invalid vocabulary: literal space is present")
    if tokenizer.pad_token_id != 0 or tokenizer.unk_token_id != 1:
        raise ValueError("Required token IDs are <pad>=0 and <unk>=1")
    if tokenizer.convert_tokens_to_ids("|") != 2:
        raise ValueError("Required word delimiter token is |=2")
    if tokenizer.word_delimiter_token != "|":
        raise ValueError("Tokenizer word_delimiter_token must be '|'")
    if len(tokenizer) != EXPECTED_CTC_VOCAB_SIZE:
        raise ValueError(
            "Bengali CTC vocabulary mismatch: "
            f"expected {EXPECTED_CTC_VOCAB_SIZE}, got {len(tokenizer)}"
        )
    if int(processor.feature_extractor.sampling_rate) != TARGET_SAMPLE_RATE:
        raise ValueError("MMS feature extractor must use 16 kHz")
    if not bool(processor.feature_extractor.do_normalize):
        raise ValueError("Expected MMS feature extractor do_normalize=True")


def build_processor(
    data_root: Path, output_dir: Path, model_name: str
) -> tuple[Wav2Vec2Processor, dict]:
    processor_dir = output_dir / "processor"
    metadata_path = processor_dir / "preprocessing_metadata.json"
    if metadata_path.is_file():
        processor = Wav2Vec2Processor.from_pretrained(processor_dir)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_processor(processor)
        if metadata.get("version") != PREPROCESSING_VERSION:
            raise RuntimeError("Existing processor uses a different preprocessing version")
        return processor, metadata

    normalized_train = []
    rejected_empty = 0
    characters = set()
    for _, _, raw in iter_text_records(data_root, "train"):
        normalized = normalize_bengali_text(raw)
        if not normalized:
            rejected_empty += 1
            continue
        normalized_train.append(normalized)
        characters.update(character for character in normalized if character != " ")
    if not normalized_train:
        raise RuntimeError("No non-empty normalized training transcripts were found")

    vocabulary = {"<pad>": 0, "<unk>": 1, "|": 2}
    for character in sorted(characters):
        if character not in vocabulary and character != " ":
            vocabulary[character] = len(vocabulary)
    if " " in vocabulary:
        raise AssertionError("Literal space must not appear in the CTC vocabulary")

    processor_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = processor_dir / "vocab.json"
    vocab_path.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        pad_token="<pad>",
        unk_token="<unk>",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
        do_lower_case=False,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    feature_extractor.return_attention_mask = True
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    validate_processor(processor)

    unknown_coverage = {}
    vocabulary_tokens = set(vocabulary)
    for split in ("validation", "test"):
        unknown = total = empty = 0
        for _, _, raw in iter_text_records(data_root, split):
            normalized = normalize_bengali_text(raw)
            if not normalized:
                empty += 1
                continue
            for character in normalized:
                if character == " ":
                    continue
                total += 1
                unknown += int(character not in vocabulary_tokens)
        unknown_coverage[split] = {
            "unknown_characters": unknown,
            "characters": total,
            "unknown_rate": unknown / max(1, total),
            "empty_after_normalization": empty,
        }

    processor.save_pretrained(processor_dir)
    vocab_bytes = json.dumps(
        vocabulary, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    metadata = {
        "version": PREPROCESSING_VERSION,
        "model_name": model_name,
        "sample_rate": TARGET_SAMPLE_RATE,
        "unicode_normalization": "NFC",
        "word_delimiter_token": "|",
        "ctc_blank_id": 0,
        "padding_id": 0,
        "unknown_id": 1,
        "vocabulary_size": len(processor.tokenizer),
        "vocabulary_sha256": hashlib.sha256(vocab_bytes).hexdigest(),
        "training_transcripts": len(normalized_train),
        "training_empty_after_normalization": rejected_empty,
        "validation_test_unknown_coverage": unknown_coverage,
        "feature_extractor": {
            "sampling_rate": int(processor.feature_extractor.sampling_rate),
            "do_normalize": bool(processor.feature_extractor.do_normalize),
            "return_attention_mask": bool(
                processor.feature_extractor.return_attention_mask
            ),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return processor, metadata


def load_audio(source, min_duration: float, max_duration: float) -> np.ndarray:
    waveform, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        raise ValueError("Audio is empty")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains non-finite samples")
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform),
            orig_freq=int(sample_rate),
            new_freq=TARGET_SAMPLE_RATE,
        ).numpy()
    waveform = waveform.astype(np.float32, copy=False)
    duration = len(waveform) / TARGET_SAMPLE_RATE
    if duration < min_duration or duration > max_duration:
        raise ValueError(
            f"Audio duration {duration:.3f}s is outside "
            f"[{min_duration}, {max_duration}]"
        )
    return waveform


def wav2vec2_output_frames(input_samples: int) -> int:
    length = int(input_samples)
    for kernel, stride in zip(
        (10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)
    ):
        length = (length - kernel) // stride + 1
    return max(0, length)


@dataclass
class DatasetOptions:
    root: Path
    split: str
    processor: Wav2Vec2Processor
    seed: int
    epoch: int
    process_index: int
    num_processes: int
    min_duration: float
    max_duration: float
    repeat: bool = False
    max_samples: int | None = None


class LocalDialectDataset(IterableDataset):
    def __init__(self, options: DatasetOptions):
        super().__init__()
        self.options = options
        self.mode = split_layout(options.root)

    def _entries(self):
        split = self.options.split
        entries = []
        if self.mode == "directories":
            split_dir = self.options.root / split
            for district in DISTRICT_TO_DIALECT:
                district_dir = split_dir / district
                wav_by_key = {
                    path.relative_to(district_dir).with_suffix("").as_posix(): path
                    for path in district_dir.rglob("*.wav")
                }
                txt_by_key = {
                    path.relative_to(district_dir).with_suffix("").as_posix(): path
                    for path in district_dir.rglob("*.txt")
                }
                if set(wav_by_key) != set(txt_by_key):
                    raise RuntimeError(f"Invalid WAV/TXT pairing in {district_dir}")
                entries.extend(
                    (district, key, wav_by_key[key], txt_by_key[key])
                    for key in sorted(wav_by_key)
                )
            return entries

        archive_path = self.options.root / f"{split}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            names = {
                info.filename for info in archive.infolist() if not info.is_dir()
            }
        grouped = {}
        for member in names:
            parts = list(PurePosixPath(member).parts)
            if parts and parts[0].lower() == split.lower():
                parts = parts[1:]
            if len(parts) < 2 or parts[0] not in DISTRICT_TO_DIALECT:
                continue
            relative = PurePosixPath(*parts[1:])
            if relative.suffix.lower() not in {".wav", ".txt"}:
                continue
            key = f"{parts[0]}/{relative.with_suffix('').as_posix()}"
            grouped.setdefault(key, {})[relative.suffix.lower()] = member
        for key in sorted(grouped):
            pair = grouped[key]
            if set(pair) != {".wav", ".txt"}:
                raise RuntimeError(f"Invalid ZIP WAV/TXT pair: {key}")
            district = key.split("/", 1)[0]
            entries.append((district, key, pair[".wav"], pair[".txt"]))
        return entries

    def _prepare(self, district, sample_id, audio_source, transcript: str):
        normalized = normalize_bengali_text(transcript)
        if not normalized:
            raise ValueError(f"Transcript became empty: {sample_id}")
        waveform = load_audio(
            audio_source, self.options.min_duration, self.options.max_duration
        )
        audio_features = self.options.processor.feature_extractor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_attention_mask=True,
        )
        labels = self.options.processor.tokenizer(normalized).input_ids
        if not labels:
            raise ValueError(f"Tokenizer produced no labels: {sample_id}")
        repeated = sum(left == right for left, right in zip(labels, labels[1:]))
        if wav2vec2_output_frames(len(waveform)) < len(labels) + repeated:
            raise ValueError(f"CTC-unalignable sample: {sample_id}")
        dialect = DISTRICT_TO_DIALECT[district]
        return {
            "input_values": np.asarray(
                audio_features["input_values"][0], dtype=np.float32
            ),
            "labels": labels,
            "dialect_label": DIALECT_TO_ID[dialect],
        }

    def __iter__(self):
        entries = self._entries()
        if not entries:
            raise RuntimeError(f"No valid entries for {self.options.split}")
        yielded = 0
        pass_index = 0
        while True:
            ordered = list(entries)
            random.Random(
                self.options.seed + self.options.epoch * 1009 + pass_index
            ).shuffle(ordered)
            archive = (
                zipfile.ZipFile(
                    self.options.root / f"{self.options.split}.zip"
                )
                if self.mode == "split-zips"
                else None
            )
            try:
                for global_index, (
                    district,
                    sample_id,
                    wav_item,
                    txt_item,
                ) in enumerate(ordered):
                    if (
                        global_index % self.options.num_processes
                        != self.options.process_index
                    ):
                        continue
                    try:
                        if archive is None:
                            transcript = txt_item.read_text(
                                encoding="utf-8-sig", errors="strict"
                            )
                            audio_source = str(wav_item)
                        else:
                            transcript = archive.read(txt_item).decode(
                                "utf-8-sig", errors="strict"
                            )
                            audio_source = io.BytesIO(archive.read(wav_item))
                        item = self._prepare(
                            district, sample_id, audio_source, transcript
                        )
                    except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
                        print(
                            f"sample_rejected id={sample_id} reason={exc}",
                            flush=True,
                        )
                        continue
                    yield item
                    yielded += 1
                    if (
                        self.options.max_samples is not None
                        and yielded >= self.options.max_samples
                    ):
                        return
            finally:
                if archive is not None:
                    archive.close()
            if not self.options.repeat:
                return
            pass_index += 1


@dataclass
class CTCDataCollator:
    processor: Wav2Vec2Processor

    def __call__(self, features):
        audio_features = [
            {"input_values": item["input_values"]} for item in features
        ]
        label_features = [{"input_ids": item["labels"]} for item in features]
        batch = self.processor.feature_extractor.pad(
            audio_features,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=True, return_tensors="pt"
        )
        target_lengths = labels_batch["attention_mask"].sum(-1).long()
        targets = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )
        batch.update(
            input_lengths=batch["attention_mask"].sum(-1).long(),
            targets=targets.long(),
            target_lengths=target_lengths,
            dialect_labels=torch.tensor(
                [item["dialect_label"] for item in features], dtype=torch.long
            ),
            dialect_label_mask=torch.ones(len(features), dtype=torch.bool),
        )
        return batch


_CTC_CONTRACT_PRINTED = False


def _validate_ctc_contract(outputs, batch, blank_id: int) -> None:
    """Fail before CTCLoss can hide a wiring or length error."""
    global _CTC_CONTRACT_PRINTED
    logits = outputs["logits"]
    ctc_blank_id = int(blank_id)
    assert ctc_blank_id == EXPECTED_CTC_BLANK_ID
    assert logits.ndim == 3
    assert logits.shape[-1] == EXPECTED_CTC_VOCAB_SIZE
    assert outputs["output_lengths"].ndim == 1
    assert outputs["output_lengths"].shape[0] == logits.shape[0]
    assert batch["target_lengths"].ndim == 1
    assert batch["target_lengths"].shape[0] == logits.shape[0]

    target_lengths = batch["target_lengths"].long()
    output_lengths = outputs["output_lengths"].long()
    if (target_lengths <= 0).any():
        raise ValueError("CTC batch contains an empty target")
    if (output_lengths <= 0).any():
        raise ValueError("CTC batch contains zero encoder output frames")
    flat_targets = torch.cat(
        [
            labels[: int(length.item())]
            for labels, length in zip(batch["targets"], target_lengths)
        ]
    ).long()
    if flat_targets.numel() != int(target_lengths.sum().item()):
        raise ValueError("CTC target lengths do not match flattened targets")
    if (flat_targets == EXPECTED_CTC_BLANK_ID).any():
        raise ValueError("CTC blank ID 0 appeared in a valid target")
    required = []
    offset = 0
    for length in target_lengths.tolist():
        target = flat_targets[offset : offset + int(length)]
        repeats = int((target[1:] == target[:-1]).sum().item())
        required.append(int(length) + repeats)
        offset += int(length)
    required_tensor = torch.tensor(required, device=output_lengths.device)
    invalid = output_lengths < required_tensor
    if invalid.any():
        bad = torch.nonzero(invalid, as_tuple=False).flatten().tolist()
        raise ValueError(
            "CTC sequence-length contract failed: "
            f"invalid_count={len(bad)} "
            f"output_lengths={output_lengths[bad].tolist()} "
            f"minimum_required={required_tensor[bad].tolist()}"
        )
    if not _CTC_CONTRACT_PRINTED:
        print("CTC blank used by loss:", ctc_blank_id, flush=True)
        print("CTC logits shape:", tuple(logits.shape), flush=True)
        print("CTC vocabulary size:", logits.shape[-1], flush=True)
        _CTC_CONTRACT_PRINTED = True


def load_balancing_loss(gate_probs, topk_indices):
    if gate_probs is None or topk_indices is None:
        device = gate_probs.device if gate_probs is not None else "cpu"
        return torch.zeros((), device=device)
    num_experts = gate_probs.shape[-1]
    importance = gate_probs.float().mean(dim=0)
    assignment = F.one_hot(
        topk_indices, num_classes=num_experts
    ).float().mean(dim=(0, 1))
    return num_experts * torch.sum(importance * assignment)


def multitask_loss(
    outputs, batch, ctc_weight, dialect_weight, balance_weight, blank_id
):
    _validate_ctc_contract(outputs, batch, blank_id)
    logits = outputs["logits"]
    log_probs = logits.float().log_softmax(-1).transpose(0, 1)
    flat_targets = torch.cat(
        [
            labels[: int(length.item())]
            for labels, length in zip(
                batch["targets"], batch["target_lengths"]
            )
        ]
    ).long()
    if (flat_targets < 0).any():
        raise ValueError("Negative target IDs cannot be passed to CTC loss")
    ctc = F.ctc_loss(
        log_probs,
        flat_targets,
        outputs["output_lengths"].long(),
        batch["target_lengths"].long(),
        blank=EXPECTED_CTC_BLANK_ID,
        reduction="mean",
        zero_infinity=True,
    )
    label_mask = batch["dialect_label_mask"].bool()
    head_loss = F.cross_entropy(
        outputs["dialect_logits"][label_mask],
        batch["dialect_labels"][label_mask],
    )
    if outputs.get("gate_probs") is not None:
        router_loss = F.nll_loss(
            outputs["gate_probs"][label_mask].clamp_min(1e-9).log(),
            batch["dialect_labels"][label_mask],
        )
        dialect = 0.5 * (head_loss + router_loss)
    else:
        dialect = head_loss
    balance = load_balancing_loss(
        outputs.get("gate_probs"), outputs.get("topk_indices")
    ).to(ctc.device)
    total = (
        ctc_weight * ctc
        + dialect_weight * dialect
        + balance_weight * balance
    )
    return total, {
        "ctc": ctc.detach(),
        "dialect": dialect.detach(),
        "balance": balance.detach(),
    }


def latest_checkpoint(output_dir: Path):
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        state_path = path / "trainer_state.json"
        if path.is_dir() and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            progress = (
                int(state.get("global_step", 0)),
                int(state.get("phase", 1)),
                -int(state.get("batch_in_phase", 0)),
                int(bool(state.get("complete", False))),
            )
            candidates.append((progress, path))
    return max(
        candidates,
        default=((0, 0, 0, 0), None),
        key=lambda item: item[0],
    )[1]


def save_checkpoint(
    accelerator,
    model,
    processor,
    output_dir,
    name,
    state,
    config,
    preprocessing,
):
    checkpoint = output_dir / name
    staging = output_dir / f".{name}.incomplete"
    if accelerator.is_main_process:
        if staging.exists():
            shutil.rmtree(staging)
        print(
            f"checkpoint_start name={name} step={state['global_step']}",
            flush=True,
        )
    accelerator.wait_for_everyone()
    accelerator.save_state(str(staging))
    if accelerator.is_main_process:
        (staging / "trainer_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        (staging / "config.json").write_text(
            json.dumps(
                OmegaConf.to_container(config, resolve=True), indent=2
            ),
            encoding="utf-8",
        )
        processor.save_pretrained(staging)
        (staging / "preprocessing_metadata.json").write_text(
            json.dumps(preprocessing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (staging / "dialect_mapping.json").write_text(
            json.dumps(
                {
                    "district_to_dialect": DISTRICT_TO_DIALECT,
                    "dialect_to_id": DIALECT_TO_ID,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if name == "checkpoint-phase-3":
            accelerator.save(
                accelerator.get_state_dict(model),
                staging / "model_state.pt",
            )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        staging.replace(checkpoint)
        print(f"checkpoint_complete name={name}", flush=True)
    accelerator.wait_for_everyone()


def prune_step_checkpoints(accelerator, output_dir: Path, keep: int):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        candidates = sorted(
            [
                path
                for path in output_dir.glob("checkpoint-step-*")
                if path.is_dir()
            ],
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        for path in candidates[:-max(1, keep)]:
            shutil.rmtree(path)
    accelerator.wait_for_everyone()


@torch.no_grad()
def validation_loss(model, loader, accelerator, config, blank_id):
    model.eval()
    total = torch.zeros((), device=accelerator.device)
    count = torch.zeros((), device=accelerator.device)
    for batch in loader:
        batch = {
            key: value.to(accelerator.device, non_blocking=True)
            for key, value in batch.items()
        }
        outputs = model(
            batch["input_values"],
            batch["attention_mask"],
            batch["input_lengths"],
        )
        loss, _ = multitask_loss(
            outputs,
            batch,
            config.loss.ctc_weight,
            config.loss.dialect_weight,
            0.0,
            blank_id,
        )
        total += loss
        count += 1
    total = accelerator.reduce(total, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    model.train()
    return float((total / count.clamp_min(1)).item())


def build_config(args, vocabulary_size, preprocessing):
    return OmegaConf.create(
        {
            "seed": args.seed,
            "experiment": "moe",
            "data": {
                "dataset": (
                    "diyalibiswas/"
                    "four-dialect-data-undersampled"
                ),
                "revision": "attached-local-mms-processor-v1",
                "min_duration": args.min_duration,
                "max_duration": args.max_duration,
                "validation_samples": args.validation_samples,
            },
            "model": {
                "pretrained_model": args.model_name,
                "num_dialects": 4,
                "num_tokens": vocabulary_size,
                "use_moe": True,
                "use_router": True,
                "use_shared_expert": True,
                "top_k": 2,
                "dropout": 0.1,
                "gradient_checkpointing": True,
            },
            "loss": {
                "ctc_weight": 1.0,
                "dialect_weight": 0.2,
                "balance_weight": 0.01,
            },
            "training": {
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "mixed_precision": "fp16",
                "head_lr": 2e-4,
                "encoder_lr": 1e-5,
                "final_encoder_lr": 5e-6,
                "weight_decay": 0.01,
                "warmup_ratio": 0.05,
                "max_grad_norm": 1.0,
                "unfrozen_top_layers": 4,
                "steps_per_phase": args.steps_per_phase,
                "checkpoint_every_steps": 100,
                "keep_last_step_checkpoints": 2,
                "log_every_steps": 200,
            },
            "preprocessing": preprocessing,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps-per-phase", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--model-name", default="facebook/mms-300m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--resume", choices=("latest",), default=None)
    parser.add_argument("--require-two-gpus", action="store_true")
    parser.add_argument("--diagnostic-mode", action="store_true")
    parser.add_argument("--diagnostic-phase1-steps", type=int, default=150)
    parser.add_argument("--diagnostic-validation-every", type=int, default=50)
    parser.add_argument("--diagnostic-delimiter-threshold", type=float, default=0.90)
    parser.add_argument("--diagnostic-delimiter-patience", type=int, default=2)
    args = parser.parse_args()
    diagnostic_mode = bool(args.diagnostic_mode)
    diagnostic_phase1_steps = min(200, max(100, int(args.diagnostic_phase1_steps)))
    diagnostic_validation_every = min(100, max(50, int(args.diagnostic_validation_every)))

    sys.path.insert(0, str(args.repo_dir / "src"))
    from asr_dialect_benchmark.modeling import BengaliDialectASR

    accelerator = Accelerator(
        gradient_accumulation_steps=16,
        mixed_precision="fp16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ],
    )
    if args.require_two_gpus and (
        accelerator.num_processes != 2 or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Expected two GPU processes; "
            f"processes={accelerator.num_processes}, "
            f"cuda={torch.cuda.is_available()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    if accelerator.is_main_process:
        _, preprocessing = build_processor(
            args.data_root, args.output_dir, args.model_name
        )
        print(json.dumps(preprocessing, ensure_ascii=False, indent=2), flush=True)
    accelerator.wait_for_everyone()
    processor = Wav2Vec2Processor.from_pretrained(
        args.output_dir / "processor"
    )
    preprocessing = json.loads(
        (
            args.output_dir
            / "processor"
            / "preprocessing_metadata.json"
        ).read_text(encoding="utf-8")
    )
    validate_processor(processor)
    blank_id = processor.tokenizer.pad_token_id
    assert blank_id == EXPECTED_CTC_BLANK_ID
    delimiter_id = processor.tokenizer.convert_tokens_to_ids("|")
    assert delimiter_id == 2
    config = build_config(args, len(processor.tokenizer), preprocessing)

    model = BengaliDialectASR(config)
    if model.ctc_head.out_features != len(processor.tokenizer):
        raise RuntimeError(
            "Fresh CTC head/tokenizer mismatch: "
            f"{model.ctc_head.out_features} vs {len(processor.tokenizer)}"
        )
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids
    ]
    optimizer = AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": config.training.encoder_lr,
                "name": "encoder",
            },
            {
                "params": head_parameters,
                "lr": config.training.head_lr,
                "name": "heads",
            },
        ],
        weight_decay=float(config.training.weight_decay),
    )
    total_updates = int(args.steps_per_phase) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(
            1, round(total_updates * float(config.training.warmup_ratio))
        ),
        num_training_steps=total_updates,
    )
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )
    collator = CTCDataCollator(processor)

    state = {
        "phase": 1,
        "phase_step": 0,
        "batch_in_phase": 0,
        "global_step": 0,
        "complete": False,
    }
    existing = latest_checkpoint(args.output_dir)
    if existing and not args.resume:
        raise RuntimeError(
            f"Output contains checkpoints: {existing}. Pass --resume latest."
        )
    resume = existing if args.resume == "latest" else None
    if resume:
        saved_config = json.loads(
            (resume / "config.json").read_text(encoding="utf-8")
        )
        saved_preprocessing = saved_config.get("preprocessing", {})
        if saved_preprocessing.get("version") != PREPROCESSING_VERSION:
            raise RuntimeError(
                "Old collapsed checkpoints are incompatible with this "
                "processor and fresh CTC head"
            )
        if (
            saved_preprocessing.get("vocabulary_sha256")
            != preprocessing["vocabulary_sha256"]
        ):
            raise RuntimeError("Resume vocabulary differs from saved processor")
        state = json.loads(
            (resume / "trainer_state.json").read_text(encoding="utf-8")
        )
        accelerator.load_state(str(resume))
        accelerator.print(
            f"Resumed {resume} at global_step={state['global_step']}"
        )

    for phase in range(int(state["phase"]), 4):
        if diagnostic_mode and phase > 1:
            accelerator.print("diagnostic_mode completed phase 1; no full training launched", flush=True)
            return
        phase_step_limit = diagnostic_phase1_steps if diagnostic_mode and phase == 1 else args.steps_per_phase
        phase_step = (
            int(state["phase_step"])
            if phase == int(state["phase"])
            else 0
        )
        accelerator.unwrap_model(model).set_phase(
            phase, int(config.training.unfrozen_top_layers)
        )
        raw_scheduler = getattr(scheduler, "scheduler", scheduler)
        if (
            phase == 3
            and raw_scheduler.base_lrs[0]
            > float(config.training.final_encoder_lr) * 1.01
        ):
            raw_scheduler.base_lrs[0] = float(
                config.training.final_encoder_lr
            )
            optimizer.param_groups[0]["lr"] *= (
                float(config.training.final_encoder_lr)
                / float(config.training.encoder_lr)
            )

        dataset = LocalDialectDataset(
            DatasetOptions(
                root=args.data_root,
                split="train",
                processor=processor,
                seed=args.seed,
                epoch=phase - 1,
                process_index=accelerator.process_index,
                num_processes=accelerator.num_processes,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                repeat=True,
            )
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        start_batch = (
            int(state["batch_in_phase"])
            if phase == int(state["phase"])
            else 0
        )
        iterator = iter(loader)
        for _ in range(start_batch):
            next(iterator)
        model.train()
        routing_buffer = []
        absolute_batch = start_batch
        accelerator.print(
            f"Starting phase={phase}/3 "
            f"phase_step={phase_step}/{phase_step_limit} "
            f"global_step={state['global_step']}",
            flush=True,
        )

        delimiter_high_streak = 0
        diagnostic_unfrozen = False
        while phase_step < phase_step_limit:
            batch = next(iterator)
            batch = {
                key: value.to(accelerator.device, non_blocking=True)
                for key, value in batch.items()
            }
            with accelerator.accumulate(model):
                outputs = model(
                    batch["input_values"],
                    batch["attention_mask"],
                    batch["input_lengths"],
                )
                loss, parts = multitask_loss(
                    outputs,
                    batch,
                    config.loss.ctc_weight,
                    config.loss.dialect_weight,
                    0.0,
                    blank_id,
                )
                if outputs.get("router_input") is not None:
                    routing_buffer.append(
                        outputs["router_input"].detach()
                    )
                if accelerator.sync_gradients and routing_buffer:
                    global_inputs = accelerator.gather(
                        torch.cat(routing_buffer, dim=0)
                    )
                    moe = accelerator.unwrap_model(model).moe
                    gate_probs, _, topk_indices = moe.route(global_inputs)
                    balance = load_balancing_loss(
                        gate_probs, topk_indices
                    )
                    loss = (
                        loss
                        + float(config.loss.balance_weight) * balance
                    )
                    parts["balance"] = balance.detach()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(config.training.max_grad_norm),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    routing_buffer.clear()
            absolute_batch += 1
            if not accelerator.sync_gradients:
                continue
            state["global_step"] += 1
            phase_step += 1
            state.update(
                phase=phase,
                phase_step=phase_step,
                batch_in_phase=absolute_batch,
                complete=False,
            )
            progress_every = diagnostic_validation_every if diagnostic_mode else 200
            diagnostic_stop = False
            if phase_step % progress_every == 0 or phase_step == phase_step_limit:
                with torch.no_grad():
                    ids = outputs["logits"].argmax(-1)
                    valid = (
                        torch.arange(ids.shape[1], device=ids.device)[None, :]
                        < outputs["output_lengths"][:, None]
                    )
                    selected = ids[valid]
                    blank_fraction = float(
                        (selected == blank_id).float().mean().item()
                    )
                    delimiter_fraction = float(
                        (selected == delimiter_id).float().mean().item()
                    )
                    diagnostic_prediction = ""
                    diagnostic_reference = ""
                    if diagnostic_mode and ids.shape[0] > 0:
                        raw_ids = ids[0, : int(outputs["output_lengths"][0].item())].tolist()
                        collapsed = []
                        previous = None
                        for token_id in raw_ids:
                            token_id = int(token_id)
                            if token_id == blank_id:
                                previous = token_id
                                continue
                            if token_id == previous:
                                continue
                            collapsed.append(token_id)
                            previous = token_id
                        diagnostic_prediction = processor.tokenizer.decode(
                            collapsed, group_tokens=False, skip_special_tokens=True
                        ).strip()
                        target_length = int(batch["target_lengths"][0].item())
                        diagnostic_reference = processor.tokenizer.decode(
                            batch["targets"][0, :target_length].tolist(),
                            group_tokens=False,
                            skip_special_tokens=True,
                        ).strip()
                accelerator.print(
                    f"progress phase={phase}/3 "
                    f"phase_step={phase_step}/{phase_step_limit} "
                    f"global_step={state['global_step']}/{total_updates} "
                    f"loss={loss.item():.4f} "
                    f"ctc={parts['ctc'].item():.4f} "
                    f"dialect={parts['dialect'].item():.4f} "
                    f"blank_fraction={blank_fraction:.4f} "
                    f"delimiter_fraction={delimiter_fraction:.4f} "
                    f"encoder_lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"head_lr={optimizer.param_groups[1]['lr']:.3e} "
                    f"diagnostic_prediction={diagnostic_prediction!r} "
                    f"diagnostic_reference={diagnostic_reference!r}",
                    flush=True,
                )
                if diagnostic_mode:
                    delimiter_high_streak = (
                        delimiter_high_streak + 1
                        if delimiter_fraction >= args.diagnostic_delimiter_threshold
                        else 0
                    )
                    if delimiter_high_streak >= max(1, args.diagnostic_delimiter_patience):
                        diagnostic_stop = True
                        accelerator.print(
                            "diagnostic_stop="
                            f"delimiter_fraction={delimiter_fraction:.4f} "
                            f"threshold={args.diagnostic_delimiter_threshold:.4f}",
                            flush=True,
                        )
            if state["global_step"] % 100 == 0:
                save_checkpoint(
                    accelerator,
                    model,
                    processor,
                    args.output_dir,
                    f"checkpoint-step-{state['global_step']:08d}",
                    state,
                    config,
                    preprocessing,
                )
                prune_step_checkpoints(accelerator, args.output_dir, 2)
            if (
                diagnostic_mode
                and not diagnostic_unfrozen
                and phase_step >= min(100, phase_step_limit)
            ):
                accelerator.unwrap_model(model).set_phase(2, int(config.training.unfrozen_top_layers))
                diagnostic_unfrozen = True
                accelerator.print("diagnostic_unfreeze=top_encoder_layers_at_step_100", flush=True)
            if diagnostic_stop:
                accelerator.print("diagnostic_mode stopping before validation/full training", flush=True)
                return

        validation_dataset = LocalDialectDataset(
            DatasetOptions(
                root=args.data_root,
                split="validation",
                processor=processor,
                seed=args.seed,
                epoch=0,
                process_index=accelerator.process_index,
                num_processes=accelerator.num_processes,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                repeat=False,
                max_samples=max(
                    1,
                    math.ceil(
                        args.validation_samples
                        / accelerator.num_processes
                    ),
                ),
            )
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=1,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        value = validation_loss(
            model,
            validation_loader,
            accelerator,
            config,
            blank_id,
        )
        state.update(
            phase=phase + 1,
            phase_step=0,
            batch_in_phase=0,
            validation_loss=value,
            complete=phase == 3,
        )
        accelerator.print(
            f"phase={phase} validation_loss={value:.4f}", flush=True
        )
        save_checkpoint(
            accelerator,
            model,
            processor,
            args.output_dir,
            f"checkpoint-phase-{phase}",
            state,
            config,
            preprocessing,
        )

    accelerator.print(
        f"Training complete at optimizer step {state['global_step']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
