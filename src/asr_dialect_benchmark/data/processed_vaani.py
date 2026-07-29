"""Map-style loader for locally attached processed Vaani Parquet shards."""

from __future__ import annotations

import io
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..tokenization.simple_tokenizer import SimpleTokenizer


class ProcessedVaaniDataset(Dataset):
    """Parquet loader with storage-locality metadata and bounded row-group caching."""

    def __init__(self, root: str, split: str, tokenizer: SimpleTokenizer | None = None, cache_row_groups: int = 4):
        self.root = Path(root)
        self.split = split
        self.tokenizer = tokenizer or SimpleTokenizer.load(self.root / "vocab.json")
        self.files = sorted((self.root / split).glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No Parquet shards found in {self.root / split}")
        self.index: List[Tuple[int, int, int]] = []
        self.durations: List[float] = []
        self.storage_groups: List[int] = []
        storage_group = 0
        for file_index, path in enumerate(self.files):
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                count = parquet.metadata.row_group(row_group).num_rows
                duration_values = parquet.read_row_group(row_group, columns=["duration"])["duration"].to_pylist()
                self.index.extend((file_index, row_group, row) for row in range(count))
                self.durations.extend(float(value) for value in duration_values)
                self.storage_groups.extend([storage_group] * count)
                storage_group += 1
        self.cache_row_groups = max(1, cache_row_groups)
        self._row_group_cache = OrderedDict()
        self._parquet_cache = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _parquet_file(self, file_index: int):
        if file_index not in self._parquet_cache:
            self._parquet_cache[file_index] = pq.ParquetFile(self.files[file_index])
            while len(self._parquet_cache) > self.cache_row_groups:
                self._parquet_cache.popitem(last=False)
        self._parquet_cache.move_to_end(file_index)
        return self._parquet_cache[file_index]

    def _record(self, index: int) -> Dict:
        file_index, row_group, row = self.index[index]
        key = (file_index, row_group)
        if key not in self._row_group_cache:
            self._row_group_cache[key] = self._parquet_file(file_index).read_row_group(row_group).to_pylist()
            while len(self._row_group_cache) > self.cache_row_groups:
                self._row_group_cache.popitem(last=False)
        self._row_group_cache.move_to_end(key)
        return self._row_group_cache[key][row]

    def __getitem__(self, index: int) -> Dict:
        record = self._record(index)
        audio, sample_rate = sf.read(io.BytesIO(record["audio_flac"]), dtype="float32", always_2d=False)
        if sample_rate != 16_000 or audio.ndim != 1:
            raise ValueError(f"Processed audio invariant failed for {record['sample_id']}")
        return {
            "input_values": torch.from_numpy(np.asarray(audio, dtype=np.float32)),
            "target": torch.tensor(self.tokenizer.encode_transcript(record["transcript"]), dtype=torch.long),
            "dialect_label": torch.tensor(record["dialect_label"], dtype=torch.long),
            "dialect_label_mask": torch.tensor(record["dialect_label_mask"], dtype=torch.bool),
            "transcript": record["transcript"],
            "sample_id": record["sample_id"],
            "speaker_id": record["speaker_id"],
            "source_district": record["source_district"],
            "residence_district": record["residence_district"],
            "dialect_group": record["dialect_group"],
        }


def processed_collate(batch: List[Dict]) -> Dict:
    audio = [item["input_values"] for item in batch]
    audio_lengths = torch.tensor([len(item) for item in audio], dtype=torch.long)
    input_values = pad_sequence(audio, batch_first=True)
    attention_mask = torch.arange(input_values.shape[1])[None, :] < audio_lengths[:, None]
    targets = [item["target"] for item in batch]
    target_lengths = torch.tensor([len(item) for item in targets], dtype=torch.long)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return {
        "input_values": input_values,
        "attention_mask": attention_mask.long(),
        "input_lengths": audio_lengths,
        "targets": padded_targets,
        "target_lengths": target_lengths,
        "dialect_labels": torch.stack([item["dialect_label"] for item in batch]),
        "dialect_label_mask": torch.stack([item["dialect_label_mask"] for item in batch]),
        **{key: [item[key] for item in batch] for key in ("transcript", "sample_id", "speaker_id", "source_district", "residence_district", "dialect_group")},
    }
