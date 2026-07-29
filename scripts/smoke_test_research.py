#!/usr/bin/env python
"""Two-GPU forward/backward and uninterrupted-versus-resumed equivalence test."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, broadcast_object_list
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from asr_dialect_benchmark.data import ProcessedVaaniDataset, processed_collate
from asr_dialect_benchmark.losses.ctc_losses import multitask_loss
from asr_dialect_benchmark.modeling import BengaliDialectASR
from asr_dialect_benchmark.training.sampler import LengthBucketBatchSampler


def optimizer_tensors(optimizer):
    optimizer = getattr(optimizer, "optimizer", optimizer)
    tensors = []
    for state in optimizer.state.values():
        for key in sorted(state):
            if torch.is_tensor(state[key]):
                tensors.append(state[key].detach().clone())
    return tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--require-two-gpus", action="store_true")
    args = parser.parse_args()
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if args.require_two_gpus and accelerator.num_processes != 2:
        raise RuntimeError(f"Expected 2 processes, found {accelerator.num_processes}")

    dataset = ProcessedVaaniDataset(args.data_dir, "train")
    config = OmegaConf.load(args.config)
    config.model.num_tokens = len(dataset.tokenizer.vocab)
    sampler = LengthBucketBatchSampler(
        dataset.durations,
        batch_size=1,
        seed=config.seed,
        storage_groups=dataset.storage_groups,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=processed_collate)
    model = BengaliDialectASR(config)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 4)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    accelerator.unwrap_model(model).set_phase(1)

    def train_step(batch):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["input_values"], batch["attention_mask"], batch["input_lengths"])
        loss, _ = multitask_loss(outputs, batch, 1.0, 0.2, 0.01)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        return loss.detach()

    iterator = iter(loader)
    first_batch = next(iterator)
    first_loss = train_step(first_batch)
    if not torch.isfinite(first_loss):
        raise RuntimeError("Non-finite first smoke-test loss")

    checkpoint_values = [tempfile.mkdtemp(prefix="vaani-resume-smoke-") if accelerator.is_main_process else None]
    broadcast_object_list(checkpoint_values)
    checkpoint = Path(checkpoint_values[0])
    accelerator.save_state(checkpoint)
    expected_random = torch.rand(4, device=accelerator.device)
    expected_batch = next(iterator)
    expected_loss = train_step(expected_batch)
    expected_parameter = next(accelerator.unwrap_model(model).ctc_head.parameters()).detach().clone()
    expected_optimizer = optimizer_tensors(optimizer)
    expected_lrs = scheduler.get_last_lr()

    accelerator.load_state(checkpoint)
    actual_random = torch.rand(4, device=accelerator.device)
    resumed_loader = accelerator.skip_first_batches(loader, 1)
    resumed_batch = next(iter(resumed_loader))
    resumed_loss = train_step(resumed_batch)
    actual_parameter = next(accelerator.unwrap_model(model).ctc_head.parameters()).detach()
    actual_optimizer = optimizer_tensors(optimizer)
    actual_lrs = scheduler.get_last_lr()

    if expected_batch["sample_id"] != resumed_batch["sample_id"]:
        raise RuntimeError("Resume did not continue with the exact next local sample")
    if not torch.equal(expected_random, actual_random):
        raise RuntimeError("Checkpoint did not exactly restore per-process RNG state")
    if not torch.equal(expected_parameter, actual_parameter):
        raise RuntimeError("Uninterrupted and resumed model parameters differ")
    if len(expected_optimizer) != len(actual_optimizer) or any(
        not torch.equal(expected, actual) for expected, actual in zip(expected_optimizer, actual_optimizer)
    ):
        raise RuntimeError("Uninterrupted and resumed optimizer states differ")
    if expected_lrs != actual_lrs:
        raise RuntimeError("Uninterrupted and resumed scheduler states differ")
    if not torch.equal(expected_loss, resumed_loss):
        raise RuntimeError("Uninterrupted and resumed losses differ")

    # Exercise the phase-2 unfreeze path under the already-created DDP reducer.
    accelerator.unwrap_model(model).set_phase(2, config.training.unfrozen_top_layers)
    phase2_batch = next(iter(accelerator.skip_first_batches(loader, 2)))
    phase2_loss = train_step(phase2_batch)
    top_layer = accelerator.unwrap_model(model).encoder.encoder.layers[-1]
    if not any(parameter.grad is not None for parameter in top_layer.parameters()):
        raise RuntimeError("Phase-2 smoke test produced no gradients in the top encoder layer")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shutil.rmtree(checkpoint)
    accelerator.wait_for_everyone()
    accelerator.print(json.dumps({
        "valid": True,
        "processes": accelerator.num_processes,
        "phase1_loss": float(first_loss.item()),
        "resumed_loss": float(resumed_loss.item()),
        "phase2_loss": float(phase2_loss.item()),
    }, indent=2))


if __name__ == "__main__":
    main()
