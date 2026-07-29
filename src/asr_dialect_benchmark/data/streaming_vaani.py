"""Direct, on-the-fly streaming of the original Vaani district datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional

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
    seed: int = 42
    epoch: int = 0
    min_duration: float = 0.5
    max_duration: float = 30.0
    shuffle_buffer: int = 1_000
    max_samples: Optional[int] = None


class VaaniStreamingDataset(IterableDataset):
    """Filter, decode, resample, and tokenize Vaani without storing audio locally."""

    def __init__(self, options: StreamingOptions, tokenizer: Optional[SimpleTokenizer] = None):
        super().__init__()
        if options.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {options.split}")
        if not options.token:
            raise ValueError("A Hugging Face token is required for gated Vaani streaming")
        self.options = options
        self.tokenizer = tokenizer or fixed_bengali_tokenizer()

    def set_epoch(self, epoch: int) -> None:
        self.options.epoch = int(epoch)

    def _source_streams(self):
        from datasets import Audio, load_dataset

        configs = list(VAANI_DISTRICT_CONFIGS)
        offset = self.options.epoch % len(configs)
        configs = configs[offset:] + configs[:offset]
        worker = get_worker_info()
        for config_index, config in enumerate(configs):
            dataset = load_dataset(
                "ARTPARK-IISc/Vaani",
                config,
                split="train",
                streaming=True,
                token=self.options.token,
                revision=self.options.revision,
            ).cast_column("audio", Audio(decode=False))
            dataset = dataset.shuffle(
                seed=self.options.seed + self.options.epoch * 1009 + config_index,
                buffer_size=self.options.shuffle_buffer,
            )
            if worker is not None and worker.num_workers > 1:
                dataset = dataset.shard(num_shards=worker.num_workers, index=worker.id)
            yield config, dataset

    def _prepare(self, row: Mapping, config: str) -> Optional[Dict]:
        transcript = normalize_bengali_text(_first(row, ("transcript", "transcription", "text", "sentence")))
        if not transcript:
            return None
        language = _first(row, ("language", "lang", "languageName", "language_name"))
        if not is_bengali_language(language):
            return None
        speaker_id = str(_first(row, ("speakerID", "speakerId", "speaker_id", "speaker"))).strip()
        if not speaker_id or speaker_split(speaker_id, self.options.seed) != self.options.split:
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
        group = DISTRICT_TO_DIALECT.get(residence_district, "")
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
