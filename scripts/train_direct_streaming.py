#!/usr/bin/env python
"""Train directly from the original gated Vaani district streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from asr_dialect_benchmark.common.constants import (
    DIALECT_MAPPING_REFERENCE,
    DIALECT_MAPPING_VERSION,
    DISTRICT_TO_DIALECT,
    VAANI_DISTRICT_CONFIGS,
)
from asr_dialect_benchmark.data import (
    StreamingOptions,
    VaaniStreamingDataset,
    fixed_bengali_tokenizer,
    processed_collate,
)
from asr_dialect_benchmark.losses.ctc_losses import load_balancing_loss, multitask_loss
from asr_dialect_benchmark.modeling import BengaliDialectASR

SECRET_RE = re.compile(
    r"^(?:hf_)?token$|access[_-]?token|auth[_-]?token|secret|password|api[_-]?key",
    re.I,
)


def sanitize(value):
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if SECRET_RE.search(str(key)) else sanitize(item))
            for key, item in value.items()
        }
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
        state_file = path / "trainer_state.json"
        if path.is_dir() and state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            progress = (
                int(state.get("global_step", 0)),
                int(state.get("phase", 1)),
                int(state.get("batch_in_phase", 0)),
                bool(state.get("complete", False)),
            )
            checkpoints.append((progress, path))
    return max(checkpoints, default=(None, None), key=lambda item: item[0] or (-1,))[1]


def save_checkpoint(accelerator, model, output_dir, name, state, config, tokenizer):
    checkpoint = output_dir / name
    staging = output_dir / f".{name}.incomplete"
    if accelerator.is_main_process:
        if staging.exists():
            shutil.rmtree(staging)
        free_gib = shutil.disk_usage(output_dir).free / (1024 ** 3)
        accelerator.print(
            f"checkpoint_start name={name} step={state['global_step']} free_disk_gib={free_gib:.2f}",
            flush=True,
        )
    accelerator.wait_for_everyone()
    started = time.monotonic()
    accelerator.save_state(str(staging))
    # Accelerate already stores resumable model weights in every checkpoint.
    # Keep a separate portable state dict only for the final evaluation checkpoint.
    if name == "checkpoint-phase-3":
        accelerator.save(accelerator.get_state_dict(model), staging / "model_state.pt")
    if accelerator.is_main_process:
        (staging / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        clean_config = sanitize(OmegaConf.to_container(config, resolve=True))
        (staging / "config.json").write_text(json.dumps(clean_config, indent=2), encoding="utf-8")
        tokenizer.save(staging / "vocab.json")
        mapping = {
            "version": DIALECT_MAPPING_VERSION,
            "reference": DIALECT_MAPPING_REFERENCE,
            "district_to_dialect": DISTRICT_TO_DIALECT,
        }
        (staging / "dialect_mapping.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if checkpoint.exists():
            if checkpoint.parent.resolve() != output_dir.resolve():
                raise RuntimeError(f"Unsafe checkpoint replacement target: {checkpoint}")
            shutil.rmtree(checkpoint)
        staging.replace(checkpoint)
        checkpoint_bytes = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
        free_gib = shutil.disk_usage(output_dir).free / (1024 ** 3)
        accelerator.print(
            f"checkpoint_complete name={name} seconds={time.monotonic() - started:.1f} "
            f"size_gib={checkpoint_bytes / (1024 ** 3):.2f} free_disk_gib={free_gib:.2f}",
            flush=True,
        )
    accelerator.wait_for_everyone()


def prune_step_checkpoints(accelerator, output_dir: Path, keep: int):
    """Bound Kaggle disk use while retaining all phase-boundary checkpoints."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        candidates = sorted(
            (path for path in output_dir.glob("checkpoint-step-*") if path.is_dir()),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        for path in candidates[:-max(1, keep)]:
            if path.parent.resolve() != output_dir.resolve() or not path.name.startswith("checkpoint-step-"):
                raise RuntimeError(f"Unsafe checkpoint cleanup target: {path}")
            shutil.rmtree(path)
    accelerator.wait_for_everyone()


def direct_training_collate(batch):
    """Keep only tensors so Accelerate can broadcast/split variable audio safely."""
    collated = processed_collate(batch)
    keys = (
        "input_values",
        "attention_mask",
        "input_lengths",
        "targets",
        "target_lengths",
        "dialect_labels",
        "dialect_label_mask",
    )
    return {key: collated[key] for key in keys}


def make_loader(config, token, tokenizer, split, epoch, max_samples=None, batch_size=None):
    allow_hf_fallback = os.environ.get("VAANI_ALLOW_HF_FALLBACK", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }
    dataset = VaaniStreamingDataset(
        StreamingOptions(
            split=split,
            token=token,
            revision=str(config.data.revision),
            allow_hf_fallback=allow_hf_fallback,
            seed=int(config.seed),
            epoch=epoch,
            min_duration=float(config.data.min_duration),
            max_duration=float(config.data.max_duration),
            shuffle_buffer=int(config.data.shuffle_buffer),
            max_samples=max_samples,
        ),
        tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or int(config.training.per_device_batch_size),
        collate_fn=direct_training_collate,
        num_workers=int(config.training.num_workers),
        pin_memory=True,
        drop_last=(batch_size or int(config.training.per_device_batch_size)) > 1,
    )


@torch.no_grad()
def validation_loss(model, loader, config, accelerator):
    model.eval()
    total = torch.zeros((), device=accelerator.device)
    count = torch.zeros((), device=accelerator.device)
    for batch in loader:
        outputs = model(batch["input_values"], batch["attention_mask"], batch["input_lengths"])
        loss, _ = multitask_loss(
            outputs, batch, config.loss.ctc_weight, config.loss.dialect_weight, 0.0
        )
        total += loss
        count += 1
    total = accelerator.reduce(total, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    model.train()
    return (total / count.clamp_min(1)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/direct_streaming.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", choices=("baseline", "moe", "top1", "no_dialect", "no_shared"), default="moe")
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Testing cap per phase")
    parser.add_argument("--validation-samples", type=int, default=None)
    parser.add_argument("--require-two-gpus", action="store_true")
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "")
    config = experiment_config(OmegaConf.load(args.config), args.experiment)
    config.output_dir = args.output_dir
    config.run_max_train_samples = args.max_train_samples
    validation_samples = args.validation_samples or int(config.data.validation_samples)
    tokenizer = fixed_bengali_tokenizer()
    config.model.num_tokens = len(tokenizer.vocab)
    vocabulary_bytes = json.dumps(tokenizer.vocab, ensure_ascii=False, sort_keys=True).encode("utf-8")
    config.dataset_fingerprints = {
        "dataset": str(config.data.dataset),
        "revision": str(config.data.revision),
        "configurations": list(VAANI_DISTRICT_CONFIGS),
        "speaker_split": f"sha256-{config.seed}-80-10-10-v1",
        "vocab_sha256": hashlib.sha256(vocabulary_bytes).hexdigest(),
        "mapping_version": DIALECT_MAPPING_VERSION,
        "streaming_layout": "district-safe-v2",
        "local_config_override": os.environ.get("VAANI_LOCAL_CONFIG", ""),
        "allow_hf_fallback": os.environ.get("VAANI_ALLOW_HF_FALLBACK", "1"),
        "local_data_mode": "paired-wav-txt-v1" if os.environ.get("VAANI_AUDIO_ROOT") else "parquet-or-hf",
        "split_policy": "supplied-directories" if os.environ.get("VAANI_AUDIO_ROOT") else "speaker-hash-v1",
    }

    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        mixed_precision=str(config.training.mixed_precision),
        dataloader_config=DataLoaderConfiguration(dispatch_batches=True, split_batches=True),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if args.require_two_gpus and (accelerator.num_processes != 2 or not torch.cuda.is_available()):
        raise RuntimeError(
            f"Expected two GPU processes, got processes={accelerator.num_processes}, cuda={torch.cuda.is_available()}"
        )
    torch.manual_seed(int(config.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = BengaliDialectASR(config)
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": config.training.encoder_lr, "name": "encoder"},
            {"params": head_parameters, "lr": config.training.head_lr, "name": "heads"},
        ],
        weight_decay=float(config.training.weight_decay),
    )
    updates_per_phase = int(config.training.estimated_optimizer_steps_per_phase)
    total_updates = updates_per_phase * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        max(1, round(total_updates * float(config.training.warmup_ratio))),
        total_updates,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    state = {"phase": 1, "batch_in_phase": 0, "global_step": 0, "complete": False}
    existing = latest_checkpoint(output_dir)
    if existing and not args.resume:
        raise RuntimeError(f"{output_dir} contains checkpoints; pass --resume latest")
    resume = existing if args.resume == "latest" else (Path(args.resume) if args.resume else None)
    if resume:
        saved_config = json.loads((resume / "config.json").read_text(encoding="utf-8"))
        current_fingerprints = OmegaConf.to_container(config.dataset_fingerprints, resolve=True)
        if saved_config.get("dataset_fingerprints") != current_fingerprints:
            raise RuntimeError("Refusing to resume with a different source revision/split/vocabulary")
        if saved_config.get("experiment") != config.experiment:
            raise RuntimeError("Refusing to resume a different experiment")
        state = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))
        accelerator.load_state(str(resume))
        accelerator.print(
            f"Resumed {resume}: phase={state['phase']} batch={state['batch_in_phase']} step={state['global_step']}"
        )

    for phase in range(int(state["phase"]), 4):
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.set_phase(phase, int(config.training.unfrozen_top_layers))
        raw_scheduler = getattr(scheduler, "scheduler", scheduler)
        if phase == 3 and raw_scheduler.base_lrs[0] > float(config.training.final_encoder_lr) * 1.01:
            raw_scheduler.base_lrs[0] = float(config.training.final_encoder_lr)
            optimizer.param_groups[0]["lr"] *= float(config.training.final_encoder_lr) / float(config.training.encoder_lr)

        train_loader = accelerator.prepare_data_loader(
            make_loader(
                config,
                token,
                tokenizer,
                "train",
                phase - 1,
                args.max_train_samples,
                batch_size=int(config.training.per_device_batch_size) * accelerator.num_processes,
            )
        )
        start_batch = int(state["batch_in_phase"]) if phase == int(state["phase"]) else 0
        phase_loader = accelerator.skip_first_batches(train_loader, start_batch) if start_batch else train_loader
        model.train()
        routing_buffer = []
        for relative_batch, batch in enumerate(phase_loader):
            absolute_batch = start_batch + relative_batch
            with accelerator.accumulate(model):
                outputs = model(batch["input_values"], batch["attention_mask"], batch["input_lengths"])
                loss, parts = multitask_loss(
                    outputs, batch, config.loss.ctc_weight, config.loss.dialect_weight, 0.0
                )
                if outputs.get("router_input") is not None:
                    routing_buffer.append(outputs["router_input"].detach())
                if accelerator.sync_gradients and routing_buffer and config.loss.balance_weight:
                    global_inputs = accelerator.gather(torch.cat(routing_buffer, dim=0))
                    moe = accelerator.unwrap_model(model).moe
                    was_training = moe.router.training if moe.router is not None else False
                    if moe.router is not None:
                        moe.router.eval()
                    gate_probs, _, topk_indices = moe.route(global_inputs)
                    if moe.router is not None:
                        moe.router.train(was_training)
                    balance = load_balancing_loss(gate_probs, topk_indices)
                    loss = loss + float(config.loss.balance_weight) * balance
                    parts["balance"] = balance.detach()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(config.training.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    routing_buffer.clear()
            if accelerator.sync_gradients:
                state["global_step"] += 1
                state.update(phase=phase, batch_in_phase=absolute_batch + 1, complete=False)
                if state["global_step"] % int(config.training.log_every_steps) == 0:
                    accelerator.print(
                        f"phase={phase} step={state['global_step']} loss={loss.item():.4f} "
                        f"ctc={parts['ctc'].item():.4f} dialect={parts['dialect'].item():.4f}"
                    )
                if state["global_step"] % int(config.training.checkpoint_every_steps) == 0:
                    save_checkpoint(
                        accelerator, model, output_dir,
                        f"checkpoint-step-{state['global_step']:08d}", state, config, tokenizer,
                    )
                    prune_step_checkpoints(
                        accelerator,
                        output_dir,
                        int(config.training.keep_last_step_checkpoints),
                    )

        validation_loader = accelerator.prepare_data_loader(
            make_loader(
                config,
                token,
                tokenizer,
                "validation",
                0,
                validation_samples,
                batch_size=int(config.training.per_device_batch_size) * accelerator.num_processes,
            )
        )
        value = validation_loss(model, validation_loader, config, accelerator)
        accelerator.print(f"phase={phase} validation_loss={value:.4f}")
        state.update(phase=phase + 1, batch_in_phase=0, validation_loss=value, complete=phase == 3)
        save_checkpoint(accelerator, model, output_dir, f"checkpoint-phase-{phase}", state, config, tokenizer)

    accelerator.print(f"Direct-stream training complete at optimizer step {state['global_step']}")


if __name__ == "__main__":
    main()
