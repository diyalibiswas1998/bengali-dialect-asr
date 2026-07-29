"""Data APIs, imported lazily so corpus building does not require PyTorch."""

__all__ = [
    "ProcessedVaaniDataset",
    "processed_collate",
    "StreamingOptions",
    "VaaniStreamingDataset",
    "fixed_bengali_tokenizer",
    "speaker_split",
]


def __getattr__(name):
    if name in {"ProcessedVaaniDataset", "processed_collate"}:
        from .processed_vaani import ProcessedVaaniDataset, processed_collate
        return {"ProcessedVaaniDataset": ProcessedVaaniDataset, "processed_collate": processed_collate}[name]
    if name in {"StreamingOptions", "VaaniStreamingDataset", "fixed_bengali_tokenizer", "speaker_split"}:
        from .streaming_vaani import (
            StreamingOptions,
            VaaniStreamingDataset,
            fixed_bengali_tokenizer,
            speaker_split,
        )
        return {
            "StreamingOptions": StreamingOptions,
            "VaaniStreamingDataset": VaaniStreamingDataset,
            "fixed_bengali_tokenizer": fixed_bengali_tokenizer,
            "speaker_split": speaker_split,
        }[name]
    raise AttributeError(name)
