import numpy as np
import torch
from torchmetrics.text import WordErrorRate, CharErrorRate


class ASREvaluation:
    def __init__(self):
        self.wer = WordErrorRate()
        self.cer = CharErrorRate()

    def compute(self, predictions, targets):
        preds = [p.strip() for p in predictions]
        targ = [t.strip() for t in targets]
        return {
            "wer": self.wer(preds, targ).item(),
            "cer": self.cer(preds, targ).item(),
        }
