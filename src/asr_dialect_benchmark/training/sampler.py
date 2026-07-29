"""Deterministic storage-local, duration-bucketed batches."""

import random
from collections import defaultdict
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    def __init__(self, durations: Sequence[float], batch_size: int, seed: int = 42, bucket_size: int = 128, drop_last: bool = False, storage_groups: Sequence[int] | None = None):
        self.durations = durations
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_size = max(bucket_size, batch_size)
        self.drop_last = drop_last
        self.storage_groups = storage_groups
        if storage_groups is not None and len(storage_groups) != len(durations):
            raise ValueError("storage_groups and durations must have equal length")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        if self.storage_groups is None:
            sorted_indices = sorted(range(len(self.durations)), key=self.durations.__getitem__)
            groups = [sorted_indices[start : start + self.bucket_size] for start in range(0, len(sorted_indices), self.bucket_size)]
        else:
            grouped = defaultdict(list)
            for index, group in enumerate(self.storage_groups):
                grouped[int(group)].append(index)
            groups = list(grouped.values())
            for group in groups:
                group.sort(key=self.durations.__getitem__)
        rng.shuffle(groups)
        batches = []
        for group in groups:
            if rng.random() < 0.5:
                group.reverse()
            batches.extend(group[start : start + self.batch_size] for start in range(0, len(group), self.batch_size))
        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]
        yield from batches

    def __len__(self) -> int:
        if self.storage_groups is None:
            length, remainder = divmod(len(self.durations), self.batch_size)
            return length if self.drop_last or not remainder else length + 1
        counts = defaultdict(int)
        for group in self.storage_groups:
            counts[int(group)] += 1
        if self.drop_last:
            return sum(count // self.batch_size for count in counts.values())
        return sum((count + self.batch_size - 1) // self.batch_size for count in counts.values())
