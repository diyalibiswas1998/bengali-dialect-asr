"""Compatibility notice for the removed manifest trainer."""


class Trainer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The legacy manifest trainer was removed because it did not implement masked labels, "
            "speaker-disjoint splits, MMS initialization, or exact resume. Use "
            "`accelerate launch scripts/train_research.py --data-dir ... --output-dir ...` instead."
        )
