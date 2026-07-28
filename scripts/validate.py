import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_dialect_benchmark.common.utils import seed_everything
from asr_dialect_benchmark.training.trainer import Trainer
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer


def main():
    seed_everything(42)
    ckpt_path = os.getenv("CKPT_PATH")
    if not ckpt_path:
        raise ValueError("Set CKPT_PATH to a checkpoint path")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("Loaded checkpoint from", ckpt_path)
    tokenizer = SimpleTokenizer()
    trainer = Trainer({"training": {"device": "cpu", "use_amp": False, "batch_size": 1, "max_epochs": 1, "lr": 1e-4, "log_dir": "logs", "output_dir": "outputs", "num_workers": 0}, "data": {"train_manifest": str(ROOT / "data" / "train.jsonl"), "val_manifest": str(ROOT / "data" / "val.jsonl"), "test_manifest": str(ROOT / "data" / "test.jsonl")}, "loss": {"use_dialect_loss": True, "use_load_balancing": True, "load_balancing_weight": 0.01}, "model": {"use_router": True, "use_shared_expert": True, "dropout": 0.1, "num_tokens": 32, "blank_index": 0, "top_k": 2}}, tokenizer=tokenizer)
    trainer.model.load_state_dict(checkpoint["model"])
    print("Validation entrypoint complete")


if __name__ == "__main__":
    main()
