#!/usr/bin/env python
"""Three-pass, fully resumable Accelerate training on processed Vaani."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from asr_dialect_benchmark.data import ProcessedVaaniDataset, processed_collate
from asr_dialect_benchmark.losses.ctc_losses import load_balancing_loss, multitask_loss
from asr_dialect_benchmark.modeling import BengaliDialectASR
from asr_dialect_benchmark.training.sampler import LengthBucketBatchSampler

SECRET_RE = re.compile(
    r"^(?:hf_)?token$|access[_-]?token|auth[_-]?token|secret|password|api[_-]?key",
    re.I,
)


def sanitize(value):
    if isinstance(value, dict):
        return {key: ("<redacted>" if SECRET_RE.search(str(key)) else sanitize(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def experiment_config(config, name: str):
    config.experiment = name
    if name == "baseline":
        config.model.use_moe = False
        config.loss.dialect_weight = 0.0
        config.loss.balance_weight = 0.0
    elif name == "top1":
        config.model.top_k = 1
    elif name == "no_dialect":
        config.loss.dialect_weight = 0.0
    elif name == "no_shared":
        config.model.use_shared_expert = False
    elif name != "moe":
        raise ValueError(f"Unknown experiment: {name}")
    return config


def latest_checkpoint(output_dir: Path):
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.search(r"(?:step|phase)-(\d+)$", path.name)
        if path.is_dir() and match:
            state_file = path / "trainer_state.json"
            if state_file.exists():
                state = json.loads(state_file.read_text(encoding="utf-8"))
                progress = (
                    int(state.get("global_step", 0)),
                    int(state.get("phase", 1)),
                    int(state.get("batch_in_phase", 0)),
                    bool(state.get("complete", False)),
                )
                checkpoints.append((progress, path))
    return max(checkpoints, default=(None, None), key=lambda item: item[0] or (-1,))[1]


def save_checkpoint(accelerator, model, output_dir: Path, name: str, state: dict, config, data_dir: Path):
    checkpoint = output_dir / name
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint))
    accelerator.save(accelerator.get_state_dict(model), checkpoint / "model_state.pt")
    if accelerator.is_main_process:
        (checkpoint / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        clean_config = sanitize(OmegaConf.to_container(config, resolve=True))
        (checkpoint / "config.json").write_text(json.dumps(clean_config, indent=2), encoding="utf-8")
        shutil.copy2(data_dir / "vocab.json", checkpoint / "vocab.json")
        shutil.copy2(data_dir / "dialect_mapping.json", checkpoint / "dialect_mapping.json")
    accelerator.wait_for_everyone()


@torch.no_grad()
def validation_loss(model, loader, config, accelerator):
    model.eval()
    total, count = torch.zeros((), device=accelerator.device), torch.zeros((), device=accelerator.device)
    for batch in loader:
        outputs = model(batch["input_values"], batch["attention_mask"], batch["input_lengths"])
        loss, _ = multitask_loss(
            outputs,
            batch,
            config.loss.ctc_weight,
            config.loss.dialect_weight,
            0.0,
        )
        total += loss
        count += 1
    total = accelerator.reduce(total, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    model.train()
    return (total / count.clamp_min(1)).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", choices=("baseline", "moe", "top1", "no_dialect", "no_shared"), default="moe")
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'")
    args = parser.parse_args()

    config = experiment_config(OmegaConf.load(args.config), args.experiment)
    config.data_dir, config.output_dir = args.data_dir, args.output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    torch.manual_seed(config.seed)
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    config.dataset_fingerprints = {
        **{split: values["content_sha256"] for split, values in dataset_metadata["splits"].items()},
        **dataset_metadata["artifact_hashes"],
        "mapping_version": dataset_metadata["mapping_version"],
    }

    train_data = ProcessedVaaniDataset(data_dir, "train")
    val_data = ProcessedVaaniDataset(data_dir, "validation", train_data.tokenizer)
    config.model.num_tokens = len(train_data.tokenizer.vocab)
    sampler = LengthBucketBatchSampler(
        train_data.durations,
        batch_size=config.training.per_device_batch_size,
        seed=config.seed,
        bucket_size=config.training.bucket_size,
        storage_groups=train_data.storage_groups,
    )
    train_loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=processed_collate, num_workers=config.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config.training.per_device_batch_size, shuffle=False, collate_fn=processed_collate, num_workers=config.training.num_workers)

    model = BengaliDialectASR(config)
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": config.training.encoder_lr, "name": "encoder"},
            {"params": head_parameters, "lr": config.training.head_lr, "name": "heads"},
        ],
        weight_decay=config.training.weight_decay,
    )
    local_batches = math.ceil(len(train_loader) / accelerator.num_processes)
    updates_per_pass = math.ceil(local_batches / config.training.gradient_accumulation_steps)
    total_updates = updates_per_pass * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, round(total_updates * config.training.warmup_ratio)), total_updates)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    state = {"phase": 1, "batch_in_phase": 0, "global_step": 0, "complete": False}
    existing_checkpoint = latest_checkpoint(output_dir)
    if existing_checkpoint and not args.resume:
        raise RuntimeError(f"{output_dir} already contains checkpoints; pass --resume latest or choose a new output directory")
    resume = existing_checkpoint if args.resume == "latest" else (Path(args.resume) if args.resume else None)
    if resume:
        saved_config = json.loads((resume / "config.json").read_text(encoding="utf-8"))
        if saved_config.get("experiment") != config.experiment:
            raise RuntimeError(f"Refusing to resume {saved_config.get('experiment')} checkpoint as {config.experiment}")
        current_fingerprints = OmegaConf.to_container(config.dataset_fingerprints, resolve=True)
        if saved_config.get("dataset_fingerprints") != current_fingerprints:
            raise RuntimeError("Refusing to resume with different processed dataset split fingerprints")
        state = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))
        accelerator.load_state(str(resume))
        accelerator.print(f"Resumed {resume} at optimizer step {state['global_step']}, phase {state['phase']}, batch {state['batch_in_phase']}")

    for phase in range(int(state["phase"]), 4):
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.set_phase(phase, config.training.unfrozen_top_layers)
        sampler.set_epoch(phase - 1)
        start_batch = int(state["batch_in_phase"]) if phase == int(state["phase"]) else 0
        # Preserve linear-decay state. At the phase-3 boundary only, halve the
        # encoder group's base and current LR; resumed phase-3 checkpoints
        # already contain this scheduler state and must not be halved again.
        raw_scheduler = getattr(scheduler, "scheduler", scheduler)
        if phase == 3 and raw_scheduler.base_lrs[0] > config.training.final_encoder_lr * 1.01:
            raw_scheduler.base_lrs[0] = config.training.final_encoder_lr
            optimizer.param_groups[0]["lr"] *= config.training.final_encoder_lr / config.training.encoder_lr
        phase_loader = accelerator.skip_first_batches(train_loader, start_batch) if start_batch else train_loader
        model.train()
        routing_buffer = []
        for relative_batch, batch in enumerate(phase_loader):
            absolute_batch = start_batch + relative_batch
            with accelerator.accumulate(model):
                outputs = model(batch["input_values"], batch["attention_mask"], batch["input_lengths"])
                loss, parts = multitask_loss(
                    outputs,
                    batch,
                    config.loss.ctc_weight,
                    config.loss.dialect_weight,
                    0.0,
                )
                if outputs.get("router_input") is not None:
                    routing_buffer.append(outputs["router_input"].detach())
                if accelerator.sync_gradients and routing_buffer and config.loss.balance_weight:
                    local_routing_inputs = torch.cat(routing_buffer, dim=0)
                    global_routing_inputs = accelerator.gather(local_routing_inputs)
                    unwrapped_moe = accelerator.unwrap_model(model).moe
                    router_was_training = unwrapped_moe.router.training if unwrapped_moe.router is not None else False
                    if unwrapped_moe.router is not None:
                        unwrapped_moe.router.eval()
                    gate_probs, _, topk_indices = unwrapped_moe.route(global_routing_inputs)
                    if unwrapped_moe.router is not None:
                        unwrapped_moe.router.train(router_was_training)
                    balance = load_balancing_loss(gate_probs, topk_indices)
                    loss = loss + config.loss.balance_weight * balance
                    parts["balance"] = balance.detach()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    routing_buffer.clear()
            if accelerator.sync_gradients:
                state["global_step"] += 1
                state.update(phase=phase, batch_in_phase=absolute_batch + 1, complete=False)
                if state["global_step"] % config.training.log_every_steps == 0:
                    accelerator.print(
                        f"phase={phase} step={state['global_step']} loss={loss.item():.4f} "
                        f"ctc={parts['ctc'].item():.4f} dialect={parts['dialect'].item():.4f} balance={parts['balance'].item():.4f}"
                    )
                if state["global_step"] % config.training.checkpoint_every_steps == 0:
                    save_checkpoint(accelerator, model, output_dir, f"checkpoint-step-{state['global_step']:08d}", state, config, data_dir)

        val_loss = validation_loss(model, val_loader, config, accelerator)
        accelerator.print(f"phase={phase} validation_loss={val_loss:.4f}")
        state.update(phase=phase + 1, batch_in_phase=0, validation_loss=val_loss, complete=phase == 3)
        save_checkpoint(accelerator, model, output_dir, f"checkpoint-phase-{phase}", state, config, data_dir)

    accelerator.print(f"Training complete at optimizer step {state['global_step']}")


if __name__ == "__main__":
    main()
