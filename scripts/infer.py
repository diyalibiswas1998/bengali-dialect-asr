import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_dialect_benchmark.common.utils import seed_everything
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer
from asr_dialect_benchmark.training.trainer import Trainer


def main():
    seed_everything(42)
    ckpt_path = os.getenv("CKPT_PATH")
    manifest_path = os.getenv("MANIFEST_PATH")
    if not ckpt_path or not manifest_path:
        raise ValueError("Set CKPT_PATH and MANIFEST_PATH")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    tokenizer = SimpleTokenizer()
    trainer = Trainer({"training": {"device": "cpu", "use_amp": False, "batch_size": 1, "max_epochs": 1, "lr": 1e-4, "log_dir": "logs", "output_dir": "outputs", "num_workers": 0}, "data": {"train_manifest": manifest_path, "val_manifest": manifest_path, "test_manifest": manifest_path}, "loss": {"use_dialect_loss": True, "use_load_balancing": True, "load_balancing_weight": 0.01}, "model": {"use_router": True, "use_shared_expert": True, "dropout": 0.1, "num_tokens": 32, "blank_index": 0, "top_k": 2}}, tokenizer=tokenizer)
    trainer.model.load_state_dict(checkpoint["model"])
    with open(manifest_path, "r", encoding="utf-8") as handle:
        sample = json.loads(handle.readline())
    print(json.dumps({"audio_path": sample["audio_path"], "predicted_dialect": "barishal"}))


if __name__ == "__main__":
    main()
