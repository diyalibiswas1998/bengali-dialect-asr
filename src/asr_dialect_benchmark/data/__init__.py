"""Data APIs, imported lazily so corpus building does not require PyTorch."""

__all__ = ["ProcessedVaaniDataset", "processed_collate"]


def __getattr__(name):
    if name in {"ProcessedVaaniDataset", "processed_collate"}:
        from .processed_vaani import ProcessedVaaniDataset, processed_collate
        return {"ProcessedVaaniDataset": ProcessedVaaniDataset, "processed_collate": processed_collate}[name]
    raise AttributeError(name)
