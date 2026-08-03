"""Direct, on-the-fly streaming of the original Vaani district datasets."""

from __future__ import annotations

import hashlib
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, Mapping, Optional, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..common.constants import DIALECT_TO_IDX, DISTRICT_TO_DIALECT, VAANI_DISTRICT_CONFIGS
from ..tokenization.simple_tokenizer import SimpleTokenizer, normalize_bengali_text
from .build_vaani import (
    _decode_audio,
    _first,
    is_bengali_language,
    normalize_district,
    wav2vec2_output_frames,
)

STREAM_COLUMNS = {
    "audio",
    "transcript", "transcription", "text", "sentence",
    "language", "lang", "languageName", "language_name",
    "speakerID", "speakerId", "speaker_id", "speaker",
    "district", "source_district", "districtName",
    "residence_district", "residenceDistrict", "stay", "residence",
    "sample_id", "id", "utterance_id", "audio_id",
}


def local_parquets_by_config(paths: Sequence[Path]) -> Dict[str, list[Path]]:
    """Match district-labelled paths without assigning flat caches to Kolkata."""
    normalized_paths = {
        path: re.sub(r"[^a-z0-9]", "", str(path).lower()) for path in paths
    }
    matches = {}
    for config in VAANI_DISTRICT_CONFIGS:
        district_key = re.sub(
            r"[^a-z0-9]", "", config.removeprefix("WestBengal_").lower()
        )
        config_matches = [
            path for path, normalized in normalized_paths.items() if district_key in normalized
        ]
        if config_matches:
            matches[config] = sorted(config_matches)
    return matches


def fixed_bengali_tokenizer() -> SimpleTokenizer:
    """Return a deterministic vocabulary without a preparatory corpus pass."""
    tokens = ["<pad>", "<unk>", " "]
    # Include the complete Bengali Unicode block so direct streaming never
    # needs a corpus-scanning vocabulary pass and never maps a retained
    # Bengali codepoint to <unk>.
    tokens.extend(map(chr, range(0x0980, 0x0A00)))
    return SimpleTokenizer(vocab={token: index for index, token in enumerate(tokens)})


def speaker_split(speaker_id: str, seed: int = 42) -> str:
    """Stable 80/10/10 assignment requiring no global metadata materialization."""
    digest = hashlib.sha256(f"{seed}|{speaker_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return "train" if bucket < 8_000 else "validation" if bucket < 9_000 else "test"


@dataclass
class StreamingOptions:
    split: str
    token: str
    revision: str
    allow_hf_fallback: bool = True
    seed: int = 42
    epoch: int = 0
    min_duration: float = 0.5
    max_duration: float = 30.0
    shuffle_buffer: int = 1_000
    max_samples: Optional[int] = None
    process_index: int = 0
    num_processes: int = 1


class VaaniStreamingDataset(IterableDataset):
    """Filter, decode, resample, and tokenize Vaani without storing audio locally."""

    def __init__(self, options: StreamingOptions, tokenizer: Optional[SimpleTokenizer] = None):
        super().__init__()
        if options.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {options.split}")
        if options.num_processes < 1 or not 0 <= options.process_index < options.num_processes:
            raise ValueError(
                f"Invalid process shard {options.process_index}/{options.num_processes}"
            )
        self.options = options
        self.tokenizer = tokenizer or fixed_bengali_tokenizer()

    def set_epoch(self, epoch: int) -> None:
        self.options.epoch = int(epoch)

    def _accept_shard(self, index: int, worker) -> bool:
        workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        shard_count = self.options.num_processes * workers
        shard_index = self.options.process_index * workers + worker_id
        return index % shard_count == shard_index

    def _paired_audio_rows(self, root: Path, worker) -> Iterator[Mapping]:
        """Yield only the 11 approved district folders from a supplied split."""
        split_dir = root / self.options.split
        if not split_dir.is_dir():
            raise RuntimeError(f"Missing local split directory: {split_dir}")

        entries = []
        counts = {}
        pair_errors = []
        for district in DISTRICT_TO_DIALECT:
            district_dir = split_dir / district
            if not district_dir.is_dir():
                raise RuntimeError(f"Missing required district directory: {district_dir}")
            wav_paths = sorted(district_dir.rglob("*.wav"))
            txt_paths = sorted(district_dir.rglob("*.txt"))
            wav_by_stem = {path.relative_to(district_dir).with_suffix(""): path for path in wav_paths}
            txt_by_stem = {path.relative_to(district_dir).with_suffix(""): path for path in txt_paths}
            missing_text = sorted(set(wav_by_stem) - set(txt_by_stem))
            missing_audio = sorted(set(txt_by_stem) - set(wav_by_stem))
            if missing_text or missing_audio:
                pair_errors.append(
                    f"{district}: missing_text={len(missing_text)} missing_audio={len(missing_audio)}"
                )
                continue
            counts[district] = len(wav_by_stem)
            entries.extend(
                (district, wav_by_stem[stem], txt_by_stem[stem]) for stem in wav_by_stem
            )
        if pair_errors:
            raise RuntimeError("Invalid WAV/TXT pairs: " + "; ".join(pair_errors))
        if not entries:
            raise RuntimeError(f"No paired WAV/TXT samples found below {split_dir}")

        random.Random(self.options.seed + self.options.epoch * 1009).shuffle(entries)
        print(
            f"[VaaniDataset] local_split={self.options.split} samples={len(entries)} "
            f"district_counts={counts}",
            flush=True,
        )
        for index, (district, wav_path, txt_path) in enumerate(entries):
            if not self._accept_shard(index, worker):
                continue
            transcript = txt_path.read_text(encoding="utf-8-sig")
            yield {
                "audio": {"path": str(wav_path)},
                "transcript": transcript,
                "language": "Bengali",
                "speakerID": wav_path.stem,
                "district": district,
                "residence_district": district,
                "sample_id": wav_path.stem,
                "_preassigned_split": self.options.split,
                "_dialect_from_district": True,
            }

    def _zipped_audio_rows(self, root: Path, worker) -> Iterator[Mapping]:
        """Yield approved district WAV/TXT pairs directly from a local split ZIP."""
        archive_path = root / f"{self.options.split}.zip"
        if not archive_path.is_file():
            raise RuntimeError(f"Missing local split archive: {archive_path}")

        with zipfile.ZipFile(archive_path) as archive:
            members = [info.filename for info in archive.infolist() if not info.is_dir()]
            by_district = {district: {"wav": {}, "txt": {}} for district in DISTRICT_TO_DIALECT}
            ignored_districts = set()
            for member in members:
                parts = list(PurePosixPath(member).parts)
                if parts and parts[0].lower() == self.options.split.lower():
                    parts = parts[1:]
                if len(parts) < 2:
                    continue
                district = parts[0]
                if district not in DISTRICT_TO_DIALECT:
                    ignored_districts.add(district)
                    continue
                relative = PurePosixPath(*parts[1:])
                suffix = relative.suffix.lower()
                if suffix not in {".wav", ".txt"}:
                    continue
                key = relative.with_suffix("").as_posix()
                by_district[district][suffix.removeprefix(".")][key] = member

            entries = []
            counts = {}
            pair_errors = []
            for district, paths in by_district.items():
                wav_keys = set(paths["wav"])
                txt_keys = set(paths["txt"])
                if wav_keys != txt_keys:
                    pair_errors.append(
                        f"{district}: missing_text={len(wav_keys - txt_keys)} "
                        f"missing_audio={len(txt_keys - wav_keys)}"
                    )
                    continue
                counts[district] = len(wav_keys)
                entries.extend(
                    (district, key, paths["wav"][key], paths["txt"][key])
                    for key in wav_keys
                )
            if pair_errors:
                raise RuntimeError("Invalid ZIP WAV/TXT pairs: " + "; ".join(pair_errors))
            if not entries:
                raise RuntimeError(f"No paired WAV/TXT samples found in {archive_path}")

            random.Random(self.options.seed + self.options.epoch * 1009).shuffle(entries)
            print(
                f"[VaaniDataset] local_zip_split={self.options.split} samples={len(entries)} "
                f"district_counts={counts} ignored_districts={sorted(ignored_districts)}",
                flush=True,
            )
            for index, (district, key, wav_member, txt_member) in enumerate(entries):
                if not self._accept_shard(index, worker):
                    continue
                transcript = archive.read(txt_member).decode("utf-8-sig")
                yield {
                    "audio": {"bytes": archive.read(wav_member)},
                    "transcript": transcript,
                    "language": "Bengali",
                    "speakerID": f"{district}/{key}",
                    "district": district,
                    "residence_district": district,
                    "sample_id": f"{district}/{key}",
                    "_preassigned_split": self.options.split,
                    "_dialect_from_district": True,
                }

    def _source_streams(self):
        import os
        from datasets import Audio, load_dataset

        configs = list(VAANI_DISTRICT_CONFIGS)
        offset = self.options.epoch % len(configs)
        configs = configs[offset:] + configs[:offset]
        worker = get_worker_info()

        audio_root = os.environ.get("VAANI_AUDIO_ROOT", "").strip()
        if audio_root:
            root = Path(audio_root)
            if not root.is_dir():
                raise RuntimeError(f"VAANI_AUDIO_ROOT does not exist: {root}")
            if (root / self.options.split).is_dir():
                stream = self._paired_audio_rows(root, worker)
            elif (root / f"{self.options.split}.zip").is_file():
                stream = self._zipped_audio_rows(root, worker)
            else:
                raise RuntimeError(
                    f"{root} has neither {self.options.split}/ nor {self.options.split}.zip"
                )
            yield "", stream
            return

        cache_dir = os.environ.get("VAANI_PARQUET_CACHE", "")
        if cache_dir and not os.path.isdir(cache_dir):
            raise RuntimeError(f"VAANI_PARQUET_CACHE does not exist: {cache_dir}")

        local_paths = sorted(Path(cache_dir).rglob("*.parquet")) if cache_dir else []
        local_by_config = local_parquets_by_config(local_paths)
        local_override = os.environ.get("VAANI_LOCAL_CONFIG", "").strip()
        if local_override:
            if local_override not in VAANI_DISTRICT_CONFIGS:
                raise RuntimeError(f"Unknown VAANI_LOCAL_CONFIG: {local_override}")
            local_by_config = {local_override: local_paths}

        def prepare_stream(dataset, shuffle_index):
            if dataset.column_names:
                selected_columns = [name for name in dataset.column_names if name in STREAM_COLUMNS]
                if "audio" not in selected_columns:
                    raise RuntimeError("Vaani source has no audio column")
                dataset = dataset.select_columns(selected_columns)
            dataset = dataset.cast_column("audio", Audio(decode=False))
            dataset = dataset.shuffle(
                seed=self.options.seed + self.options.epoch * 1009 + shuffle_index,
                buffer_size=self.options.shuffle_buffer,
            )
            if worker is not None and worker.num_workers > 1:
                dataset = dataset.shard(num_shards=worker.num_workers, index=worker.id)
            return dataset

        # A flat cache cannot safely be called Kolkata. Load it once and require
        # every accepted row to carry its own district metadata.
        if local_paths and not local_by_config:
            print(
                f"[VaaniDataset] Loading {len(local_paths)} consolidated local Parquet files once; "
                "source_district must be present in each row",
                flush=True,
            )
            dataset = load_dataset(
                "parquet", data_files=[str(path) for path in local_paths], split="train", streaming=True
            )
            district_columns = {"district", "source_district", "districtName"}
            if dataset.column_names and not district_columns.intersection(dataset.column_names):
                raise RuntimeError(
                    "Flat local Parquet cache has no district column. Set VAANI_LOCAL_CONFIG "
                    "only if every file belongs to one known district."
                )
            yield "", prepare_stream(dataset, 0)
            return

        for config_index, config in enumerate(configs):
            dataset = None
            matching = local_by_config.get(config, [])
            if matching:
                try:
                    print(
                        f"[VaaniDataset] Loading {len(matching)} local Parquet files for {config}",
                        flush=True,
                    )
                    dataset = load_dataset(
                        "parquet", data_files=[str(path) for path in matching], split="train", streaming=True
                    )
                except Exception as exc:
                    print(
                        f"[VaaniDataset] Could not load local Parquet files: {exc}. Falling back to HF stream.",
                        flush=True,
                    )
                    dataset = None

            if dataset is None:
                if not self.options.allow_hf_fallback:
                    raise RuntimeError(
                        f"No local Parquet files matched {config} and Hugging Face fallback is disabled"
                    )
                if not self.options.token:
                    raise RuntimeError(
                        f"No local Parquet files matched {config}; HF_TOKEN is required only for fallback"
                    )
                dataset = load_dataset(
                    "ARTPARK-IISc/Vaani",
                    config,
                    split="train",
                    streaming=True,
                    token=self.options.token,
                    revision=self.options.revision,
                )

            yield config, prepare_stream(dataset, config_index)

    def _prepare(self, row: Mapping, config: str) -> Optional[Dict]:
        transcript = normalize_bengali_text(_first(row, ("transcript", "transcription", "text", "sentence")))
        if not transcript:
            return None
        language = _first(row, ("language", "lang", "languageName", "language_name"))
        if not is_bengali_language(language):
            return None
        speaker_id = str(_first(row, ("speakerID", "speakerId", "speaker_id", "speaker"))).strip()
        preassigned_split = str(row.get("_preassigned_split", "")).strip()
        if not speaker_id:
            return None
        if preassigned_split:
            if preassigned_split != self.options.split:
                return None
        elif speaker_split(speaker_id, self.options.seed) != self.options.split:
            return None
        source_district = normalize_district(
            _first(row, ("district", "source_district", "districtName"))
        ) or normalize_district(config)
        if not source_district:
            return None
        residence_district = ""
        for field in ("residence_district", "residenceDistrict", "stay", "residence"):
            residence_district = normalize_district(row.get(field))
            if residence_district:
                break
        try:
            audio, sample_rate = _decode_audio(row.get("audio"))
        except Exception:
            return None
        duration = len(audio) / float(sample_rate)
        if not self.options.min_duration <= duration <= self.options.max_duration:
            return None
        target = self.tokenizer.encode_transcript(transcript)
        minimum_ctc_frames = len(target) + sum(left == right for left, right in zip(target, target[1:]))
        if not target or wav2vec2_output_frames(len(audio)) < minimum_ctc_frames:
            return None
        label_district = source_district if row.get("_dialect_from_district") else residence_district
        group = DISTRICT_TO_DIALECT.get(label_district, "")
        label = DIALECT_TO_IDX[group] if group else -100
        source_id = str(_first(row, ("sample_id", "id", "utterance_id", "audio_id"), ""))
        if not source_id:
            source_id = hashlib.sha256(audio.tobytes()).hexdigest()
        sample_id = hashlib.sha256(
            f"vaani-stream|{config}|{speaker_id}|{source_id}".encode("utf-8")
        ).hexdigest()
        return {
            "input_values": torch.from_numpy(audio),
            "target": torch.tensor(target, dtype=torch.long),
            "dialect_label": torch.tensor(label, dtype=torch.long),
            "dialect_label_mask": torch.tensor(label >= 0, dtype=torch.bool),
            "transcript": transcript,
            "sample_id": sample_id,
            "speaker_id": speaker_id,
            "source_district": source_district,
            "residence_district": residence_district,
            "dialect_group": group,
        }

    def __iter__(self) -> Iterator[Dict]:
        accepted = 0
        for config, stream in self._source_streams():
            for row in stream:
                sample = self._prepare(row, config)
                if sample is None:
                    continue
                yield sample
                accepted += 1
                if self.options.max_samples is not None and accepted >= self.options.max_samples:
                    return
        if accepted == 0:
            raise RuntimeError(
                f"No valid {self.options.split} samples were found. For a flat single-district "
                "cache set VAANI_LOCAL_CONFIG; consolidated caches must include district metadata."
            )
