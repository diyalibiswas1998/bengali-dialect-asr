#!/usr/bin/env python
"""One-batch two-GPU smoke test against the original Vaani stream."""

import argparse
import os

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf

from asr_dialect_benchmark.data import fixed_bengali_tokenizer
from asr_dialect_benchmark.losses.ctc_losses import multitask_loss
from asr_dialect_benchmark.modeling import BengaliDialectASR
from train_direct_streaming import make_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/direct_streaming.yaml")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--require-two-gpus", action="store_true")
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        raise RuntimeError(f"Missing {args.token_env}")
    config = OmegaConf.load(args.config)
    tokenizer = fixed_bengali_tokenizer()
    config.model.num_tokens = len(tokenizer.vocab)
    accelerator = Accelerator(
        mixed_precision=str(config.training.mixed_precision),
        dataloader_config=DataLoaderConfiguration(dispatch_batches=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if args.require_two_gpus and (accelerator.num_processes != 2 or not torch.cuda.is_available()):
        raise RuntimeError(
            f"Expected two GPU processes, got processes={accelerator.num_processes}, cuda={torch.cuda.is_available()}"
        )
    loader = make_loader(config, token, tokenizer, "train", 0, max_samples=8)
    model = BengaliDialectASR(config)
    model.set_phase(1, int(config.training.unfrozen_top_layers))
    model, loader = accelerator.prepare(model, loader)
    batch = next(iter(loader))
    outputs = model(batch["input_values"], batch["attention_mask"], batch["input_lengths"])
    loss, parts = multitask_loss(
        outputs,
        batch,
        config.loss.ctc_weight,
        config.loss.dialect_weight,
        config.loss.balance_weight,
    )
    accelerator.backward(loss)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite direct-stream smoke loss")
    accelerator.print(
        f"Direct-stream smoke passed: processes={accelerator.num_processes} "
        f"loss={loss.item():.4f} ctc={parts['ctc'].item():.4f}"
    )


if __name__ == "__main__":
    main()
