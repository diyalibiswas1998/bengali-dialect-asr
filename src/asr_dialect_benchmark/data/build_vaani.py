"""Build the reproducible local Bengali Vaani research corpus."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import soundfile as sf

from ..common.constants import (
    BOUNDARY_DISTRICTS,
    DIALECT_MAPPING_REFERENCE,
    DIALECT_MAPPING_VERSION,
    DIALECT_TO_IDX,
    DISTRICT_TO_DIALECT,
    VAANI_DISTRICT_CONFIGS,
)
from ..tokenization.simple_tokenizer import SimpleTokenizer, normalize_bengali_text

SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("audio_flac", pa.binary()),
        ("duration", pa.float32()),
        ("transcript", pa.string()),
        ("speaker_id", pa.string()),
        ("source_district", pa.string()),
        ("residence_district", pa.string()),
        ("dialect_group", pa.string()),
        ("dialect_label", pa.int8()),
        ("dialect_label_mask", pa.bool_()),
    ]
)

DISTRICT_KEYS = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in DISTRICT_TO_DIALECT}
BENGALI_LANGUAGE_KEYS = {"bengali", "bangla", "ben", "bn", "বাংলা", "বাংলা ভাষা"}


@dataclass
class BuildOptions:
    output_dir: str
    source: str = "auto"
    token: Optional[str] = None
    seed: int = 42
    min_duration: float = 0.5
    max_duration: float = 30.0
    shard_size: int = 2_000
    max_samples: Optional[int] = None
    keep_temporary: bool = False
    allow_main_fallback: bool = False
    resume_staging: Optional[str] = None


def update_record_fingerprint(hasher, record: Mapping) -> None:
    """Hash every field that can change model inputs, labels, or split identity."""
    audio_hash = record.get("audio_hash") or hashlib.sha256(record["audio_flac"]).hexdigest()
    payload = {
        "sample_id": record["sample_id"],
        "audio_sha256": audio_hash,
        "transcript": record["transcript"],
        "speaker_id": record["speaker_id"],
        "source_district": record["source_district"],
        "residence_district": record["residence_district"],
        "dialect_group": record["dialect_group"],
        "dialect_label": int(record["dialect_label"]),
        "dialect_label_mask": bool(record["dialect_label_mask"]),
    }
    hasher.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hasher.update(b"\n")


def normalize_district(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("WestBengal_"):
        text = text.split("_", 1)[1]
    # Residence fields commonly look like "Alipurduar(20)".
    text = text.split("(", 1)[0]
    key = re.sub(r"[^a-z0-9]", "", text.lower())
    return DISTRICT_KEYS.get(key, "")


def is_bengali_language(value: object) -> bool:
    return str(value or "").strip().lower() in BENGALI_LANGUAGE_KEYS


def _first(row: Mapping, names: Iterable[str], default=""):
    for name in names:
        value = row.get(name)
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            continue
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null"}:
            return value
    return default


def _decode_audio(value: object, target_rate: int = 16_000) -> Tuple[np.ndarray, int]:
    if isinstance(value, Mapping) and value.get("array") is not None:
        audio = np.asarray(value["array"], dtype=np.float32)
        sample_rate = int(value.get("sampling_rate") or target_rate)
    else:
        raw = value.get("bytes") if isinstance(value, Mapping) else None
        path = value.get("path") if isinstance(value, Mapping) else None
        source = io.BytesIO(raw) if raw else path
        if not source:
            raise ValueError("audio has neither bytes, array, nor path")
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        # soundfile is [samples, channels]; decoded dataset arrays may be
        # [channels, samples]. Infer the small channel dimension.
        channel_axis = 0 if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0] else 1
        audio = audio.mean(axis=channel_axis)
    elif audio.ndim != 1:
        raise ValueError(f"unsupported audio rank: {audio.ndim}")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("audio is empty or non-finite")
    if sample_rate != target_rate:
        new_length = round(len(audio) * target_rate / sample_rate)
        if new_length <= 0:
            raise ValueError("invalid resampled length")
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        sample_rate = target_rate
    return np.clip(audio, -1.0, 1.0).astype(np.float32), sample_rate


def _to_flac(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def wav2vec2_output_frames(input_samples: int) -> int:
    length = int(input_samples)
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        length = (length - kernel) // stride + 1
    return max(0, length)


def parse_row(row: Mapping, config: str, options: BuildOptions) -> Tuple[Optional[Dict], str]:
    transcript = normalize_bengali_text(_first(row, ("transcript", "transcription", "text", "sentence")))
    if not transcript:
        return None, "empty_transcript"

    language = _first(row, ("language", "lang", "languageName", "language_name"))
    # The transcription repository's Bengali configuration may omit language.
    if config != "Bengali" and not is_bengali_language(language):
        return None, "non_bengali"
    if config == "Bengali" and language and not is_bengali_language(language):
        return None, "non_bengali"

    speaker_id = str(_first(row, ("speakerID", "speakerId", "speaker_id", "speaker"))).strip()
    if not speaker_id:
        return None, "missing_speaker"

    source_district = normalize_district(
        _first(row, ("district", "source_district", "districtName"))
    ) or normalize_district(config)
    if not source_district:
        return None, "unknown_source_district"
    residence_district = ""
    for field in ("residence_district", "residenceDistrict", "stay", "residence"):
        residence_district = normalize_district(row.get(field))
        if residence_district:
            break

    try:
        audio, sample_rate = _decode_audio(row.get("audio"))
    except Exception:
        return None, "audio_decode"
    duration = len(audio) / float(sample_rate)
    if not options.min_duration <= duration <= options.max_duration:
        return None, "duration"
    minimum_ctc_frames = len(transcript) + sum(
        left == right for left, right in zip(transcript, transcript[1:])
    )
    if wav2vec2_output_frames(len(audio)) < minimum_ctc_frames:
        return None, "ctc_unalignable"

    flac = _to_flac(audio, sample_rate)
    audio_hash = hashlib.sha256(flac).hexdigest()
    source_id = str(_first(row, ("sample_id", "id", "utterance_id", "audio_id"), audio_hash))
    sample_id = hashlib.sha256(f"vaani|{source_district}|{speaker_id}|{source_id}|{audio_hash}".encode()).hexdigest()
    group = DISTRICT_TO_DIALECT.get(residence_district, "")
    label = DIALECT_TO_IDX[group] if group else -100
    return {
        "sample_id": sample_id,
        "audio_flac": flac,
        "audio_hash": audio_hash,
        "duration": np.float32(duration),
        "transcript": transcript,
        "speaker_id": speaker_id,
        "source_district": source_district,
        "residence_district": residence_district,
        "dialect_group": group,
        "dialect_label": label,
        "dialect_label_mask": label >= 0,
    }, "accepted"


class ShardWriter:
    def __init__(self, directory: Path, schema: pa.Schema, shard_size: int, row_group_size: int = 64):
        self.directory, self.schema, self.shard_size = directory, schema, shard_size
        self.row_group_size = row_group_size
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        existing = sorted(self.directory.glob("part-*.parquet"))
        self.index = len(existing)

    def add(self, record: Dict) -> None:
        self.buffer.append({name: record[name] for name in self.schema.names})
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        pq.write_table(
            table,
            self.directory / f"part-{self.index:05d}.parquet",
            compression="zstd",
            row_group_size=self.row_group_size,
        )
        self.buffer.clear()
        self.index += 1

    def close(self) -> None:
        self.flush()


def _hf_streams(options: BuildOptions) -> Iterator[Tuple[str, Iterable[Mapping]]]:
    from datasets import Audio, load_dataset

    kwargs = {"split": "train", "streaming": True}
    if options.token:
        kwargs["token"] = options.token

    if options.source in {"auto", "transcription"}:
        try:
            dataset = load_dataset("ARTPARK-IISc/Vaani-transcription-part", "Bengali", **kwargs)
            dataset = dataset.cast_column("audio", Audio(decode=False))
            first = next(iter(dataset))
        except Exception as exc:
            raise RuntimeError(
                "Could not inspect the Bengali transcription configuration. Fix authentication/network access; "
                "use --source main only when intentionally processing the full raw corpus."
            ) from exc
        fields = set(first)
        requirements = {
            "audio": {"audio"},
            "transcript": {"transcript", "transcription", "text", "sentence"},
            "district": {"district", "source_district", "districtName"},
            "speaker": {"speakerID", "speakerId", "speaker_id", "speaker"},
            "duration": {"duration", "audio_duration", "duration_seconds"},
            "residence": {"residence_district", "residenceDistrict", "stay", "residence"},
        }
        missing = [name for name, alternatives in requirements.items() if not fields & alternatives]
        if not missing:
            print("Using ARTPARK-IISc/Vaani-transcription-part, configuration Bengali.")
            dataset = load_dataset("ARTPARK-IISc/Vaani-transcription-part", "Bengali", **kwargs)
            yield "Bengali", dataset.cast_column("audio", Audio(decode=False))
            return
        if options.source == "transcription" or not options.allow_main_fallback:
            raise RuntimeError(
                f"Bengali transcription configuration lacks required fields: {missing}. "
                "Pass --allow-main-fallback to acknowledge processing all 11 raw district configurations."
            )
        print(f"Transcription configuration lacks {missing}; explicit main-corpus fallback was authorized.")

    for config in VAANI_DISTRICT_CONFIGS:
        print(f"Streaming source configuration {config} ...")
        dataset = load_dataset("ARTPARK-IISc/Vaani", config, **kwargs)
        yield config, dataset.cast_column("audio", Audio(decode=False))


def assign_speaker_splits(speaker_stats: Mapping[str, Counter], seed: int = 42) -> Dict[str, str]:
    """Greedily match 80/10/10 totals and dominant dialect/district strata."""
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    total = sum(sum(counter.values()) for counter in speaker_stats.values())
    strata_totals = Counter()
    for counter in speaker_stats.values():
        strata_totals.update(counter)
    used_total = Counter()
    used_strata = defaultdict(Counter)
    rng = random.Random(seed)
    speakers = list(speaker_stats)
    rng.shuffle(speakers)
    speakers.sort(key=lambda speaker: -sum(speaker_stats[speaker].values()))
    assignments = {}
    for speaker in speakers:
        counter = speaker_stats[speaker]
        size = sum(counter.values())
        best_split, best_score = None, None
        for split, ratio in ratios.items():
            total_deficit = ratio * total - used_total[split]
            stratum_deficit = sum(
                count * (ratio * strata_totals[stratum] - used_strata[split][stratum])
                / max(1, strata_totals[stratum])
                for stratum, count in counter.items()
            )
            score = total_deficit / max(1, total) + stratum_deficit / max(1, size)
            if best_score is None or score > best_score:
                best_split, best_score = split, score
        assignments[speaker] = best_split
        used_total[best_split] += size
        used_strata[best_split].update(counter)
    # Tiny smoke-test subsets can otherwise place every speaker in train.
    # Guarantee non-empty splits without affecting normal full-corpus builds.
    for missing_split in (split for split in ratios if split not in assignments.values()):
        donor_counts = Counter(assignments.values())
        candidates = [
            speaker for speaker, split in assignments.items() if donor_counts[split] > 1
        ]
        if not candidates:
            raise RuntimeError("At least three distinct speakers are required for 80/10/10 splits")
        moved = min(candidates, key=lambda speaker: sum(speaker_stats[speaker].values()))
        assignments[moved] = missing_split
    return assignments


def _stats_template():
    return {
        "samples": 0,
        "hours": 0.0,
        "speakers": set(),
        "district_samples": Counter(),
        "district_hours": Counter(),
        "district_speakers": defaultdict(set),
        "dialect_samples": Counter(),
        "dialect_hours": Counter(),
        "dialect_speakers": defaultdict(set),
        "labeled": 0,
    }


def build_processed_corpus(options: BuildOptions) -> Path:
    output = Path(options.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if options.resume_staging:
        staging = Path(options.resume_staging).resolve()
        if not staging.is_dir() or staging.parent != output.parent:
            raise ValueError("Resume staging directory must exist beside the requested output directory")
    else:
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent))
    print(f"Build staging directory: {staging}")
    temporary = staging / "_validated"
    temp_schema = SCHEMA.append(pa.field("audio_hash", pa.string()))
    writer = ShardWriter(temporary, temp_schema, options.shard_size)
    rejection_counts, seen_audio, seen_ids = Counter(), set(), set()
    speaker_stats = defaultdict(Counter)
    accepted = 0
    used_source_configs = []
    ingestion_marker = staging / "_INGESTION_COMPLETE.json"
    try:
        if options.resume_staging and list(temporary.glob("part-*.parquet")):
            print("Reconstructing deduplication and speaker state from existing validated shards ...")
            existing_dataset = pads.dataset(temporary, format="parquet")
            for batch in existing_dataset.to_batches(batch_size=512):
                for record in batch.to_pylist():
                    seen_audio.add(record["audio_hash"])
                    seen_ids.add(record["sample_id"])
                    stratum = (record["dialect_label"], record["source_district"])
                    speaker_stats[record["speaker_id"]][stratum] += 1
                    accepted += 1
            print(f"Recovered {accepted:,} validated samples; source streaming will skip duplicates.")
        if ingestion_marker.exists():
            marker_data = json.loads(ingestion_marker.read_text(encoding="utf-8"))
            used_source_configs = marker_data.get("source_configurations", [])
            source_streams = ()
            print("Validated source ingestion is already complete; restarting deterministic repartition only.")
        else:
            source_streams = _hf_streams(options)
        stop = False
        for config, stream in source_streams:
            used_source_configs.append(config)
            for row in stream:
                record, status = parse_row(row, config, options)
                if record is None:
                    rejection_counts[status] += 1
                    continue
                if record["audio_hash"] in seen_audio or record["sample_id"] in seen_ids:
                    rejection_counts["duplicate"] += 1
                    continue
                seen_audio.add(record["audio_hash"])
                seen_ids.add(record["sample_id"])
                stratum = (record["dialect_label"], record["source_district"])
                speaker_stats[record["speaker_id"]][stratum] += 1
                writer.add(record)
                accepted += 1
                if accepted % 1000 == 0:
                    print(f"Accepted {accepted:,} unique valid samples ...")
                if options.max_samples and accepted >= options.max_samples:
                    stop = True
                    break
            if stop:
                break
        writer.close()
        if not accepted:
            raise RuntimeError("No valid samples were found")
        if not ingestion_marker.exists():
            ingestion_marker.write_text(
                json.dumps({"accepted": accepted, "source_configurations": used_source_configs}, indent=2),
                encoding="utf-8",
            )

        assignments = assign_speaker_splits(speaker_stats, options.seed)
        # Repartitioning is deterministic and cheap relative to source audio
        # processing, so restart only these generated children on resume.
        for child_name in ("train", "validation", "test", "manifests"):
            child = staging / child_name
            if child.exists():
                shutil.rmtree(child)
        split_writers = {split: ShardWriter(staging / split, SCHEMA, options.shard_size) for split in ("train", "validation", "test")}
        stats = {split: _stats_template() for split in split_writers}
        split_hashes = {split: hashlib.sha256() for split in split_writers}
        train_transcripts = []
        manifest_dir = staging / "manifests"
        manifest_dir.mkdir()
        manifest_handles = {
            split: (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in split_writers
        }
        try:
            dataset = pads.dataset(temporary, format="parquet")
            for batch in dataset.to_batches(batch_size=512):
                for record in batch.to_pylist():
                    split = assignments[record["speaker_id"]]
                    record.pop("audio_hash", None)
                    split_writers[split].add(record)
                    update_record_fingerprint(split_hashes[split], record)
                    manifest = {key: value for key, value in record.items() if key != "audio_flac"}
                    manifest_handles[split].write(json.dumps(manifest, ensure_ascii=False) + "\n")
                    item_stats = stats[split]
                    item_stats["samples"] += 1
                    item_stats["hours"] += record["duration"] / 3600.0
                    item_stats["speakers"].add(record["speaker_id"])
                    district = record["source_district"]
                    dialect = record["dialect_group"] or "unlabeled"
                    item_stats["district_samples"][district] += 1
                    item_stats["district_hours"][district] += record["duration"] / 3600.0
                    item_stats["district_speakers"][district].add(record["speaker_id"])
                    item_stats["dialect_samples"][dialect] += 1
                    item_stats["dialect_hours"][dialect] += record["duration"] / 3600.0
                    item_stats["dialect_speakers"][dialect].add(record["speaker_id"])
                    item_stats["labeled"] += int(record["dialect_label_mask"])
                    if split == "train":
                        train_transcripts.append(record["transcript"])
        finally:
            for handle in manifest_handles.values():
                handle.close()
        for split_writer in split_writers.values():
            split_writer.close()

        tokenizer = SimpleTokenizer()
        tokenizer.fit_from_transcripts(train_transcripts)
        tokenizer.save(staging / "vocab.json")
        mapping = {
            "version": DIALECT_MAPPING_VERSION,
            "reference": DIALECT_MAPPING_REFERENCE,
            "provisional_geographic_proxy": True,
            "district_to_dialect": DISTRICT_TO_DIALECT,
            "dialect_to_label": DIALECT_TO_IDX,
            "boundary_districts": BOUNDARY_DISTRICTS,
        }
        (staging / "dialect_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        artifact_hashes = {
            "vocab_sha256": hashlib.sha256((staging / "vocab.json").read_bytes()).hexdigest(),
            "dialect_mapping_sha256": hashlib.sha256((staging / "dialect_mapping.json").read_bytes()).hexdigest(),
        }
        serializable_stats = {}
        for split, values in stats.items():
            districts = {
                name: {
                    "samples": count,
                    "hours": round(values["district_hours"][name], 6),
                    "speakers": len(values["district_speakers"][name]),
                }
                for name, count in sorted(values["district_samples"].items())
            }
            dialects = {
                name: {
                    "samples": count,
                    "hours": round(values["dialect_hours"][name], 6),
                    "speakers": len(values["dialect_speakers"][name]),
                }
                for name, count in sorted(values["dialect_samples"].items())
            }
            serializable_stats[split] = {
                "samples": values["samples"],
                "labeled": values["labeled"],
                "hours": round(values["hours"], 6),
                "speakers": len(values["speakers"]),
                "districts": districts,
                "dialects": dialects,
                "content_sha256": split_hashes[split].hexdigest(),
            }
        metadata = {
            "format_version": 1,
            "source": "ARTPARK-IISc/Vaani and/or Vaani-transcription-part",
            "source_configurations": used_source_configs,
            "source_license_must_be_reviewed": True,
            "build_options": {**asdict(options), "token": None},
            "mapping_version": DIALECT_MAPPING_VERSION,
            "artifact_hashes": artifact_hashes,
            "schema": [field.name for field in SCHEMA],
            "rejections": dict(rejection_counts),
            "splits": serializable_stats,
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (staging / "SOURCE_AND_LICENSE.md").write_text(
            "# Source and license\n\nDerived from ARTPARK-IISc Vaani. Preserve the upstream attribution and "
            "verify the current upstream dataset terms before redistribution. Dialect labels are provisional "
            f"geographic proxies ({DIALECT_MAPPING_VERSION}), not linguistic ground truth.\n",
            encoding="utf-8",
        )
        if not options.keep_temporary:
            shutil.rmtree(temporary)
            ingestion_marker.unlink(missing_ok=True)
        staging.replace(output)
        return output
    except Exception as exc:
        raise RuntimeError(
            f"Build stopped; validated staging was preserved at {staging}. "
            "Retry with --resume-staging pointing to that directory."
        ) from exc
