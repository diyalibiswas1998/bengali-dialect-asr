#!/usr/bin/env python
"""Check research checkpoint completeness, resume metadata, and secrets."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-processes", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.checkpoint)
    required = ["trainer_state.json", "config.json", "vocab.json", "dialect_mapping.json", "model_state.pt"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint; missing {missing}")
    for pattern in ("optimizer*", "scheduler*", "random_states*"):
        if not list(root.glob(pattern)):
            raise FileNotFoundError(f"Accelerate state missing {pattern}")
    random_states = list(root.glob("random_states*"))
    if len(random_states) < args.expected_processes:
        raise RuntimeError(f"Expected RNG state for {args.expected_processes} processes, found {len(random_states)}")

    state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "dialect_mapping.json").read_text(encoding="utf-8"))
    vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    assert isinstance(state.get("global_step"), int) and state["global_step"] >= 0
    assert isinstance(state.get("phase"), int) and 1 <= state["phase"] <= 4
    assert isinstance(state.get("batch_in_phase"), int) and state["batch_in_phase"] >= 0
    assert config["model"]["pretrained_model"] == "facebook/mms-300m"
    assert config["model"]["num_tokens"] == len(vocab)
    assert mapping["version"] == "west-bengal-proxy-v1"
    assert {"train", "validation", "test", "vocab_sha256", "dialect_mapping_sha256", "mapping_version"} <= set(config["dataset_fingerprints"])

    secret = re.compile(rb"hf_[A-Za-z0-9]{20,}")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            overlap = b""
            while chunk := handle.read(8 * 1024 * 1024):
                block = overlap + chunk
                if secret.search(block):
                    raise RuntimeError(f"Possible Hugging Face secret found in {path}")
                overlap = block[-64:]
    print(json.dumps({"valid": True, "checkpoint": str(root), "state": state, "random_state_files": len(random_states)}, indent=2))


if __name__ == "__main__":
    main()
