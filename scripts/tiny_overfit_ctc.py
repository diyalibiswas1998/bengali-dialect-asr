#!/usr/bin/env python
"""Plain MMS-CTC 32-sample overfit test for the Bengali tokenizer.

The script deliberately excludes MoE, routing, dialect loss, augmentation, and
distributed training.  It requires a manually verified CSV manifest so a
transcript/audio mismatch cannot be mistaken for a model failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from ctc_collapse_diagnostics import (
    EXPECTED_BLANK_ID,
    EXPECTED_DELIMITER_ID,
    EXPECTED_VOCAB_SIZE,
    TARGET_SAMPLE_RATE,
    _decode,
    _load_processor,
    edit_distance,
    read_audio,
)


def grad_norm(parameters) -> float:
    import torch

    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        if not torch.isfinite(value).all():
            return float("nan")
        total += float(value.pow(2).sum().item())
    return total ** 0.5


def feature_output_lengths(model, input_lengths):
    encoder = getattr(model, "wav2vec2", model)
    return encoder._get_feat_extract_output_lengths(input_lengths).long()


def read_manifest(path: Path, manually_verified: bool) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The tiny manifest is empty")
    if len(rows) < 20 or len(rows) > 50:
        raise ValueError(f"Tiny overfit manifest must contain 20-50 rows, got {len(rows)}")
    if not manually_verified:
        raise RuntimeError("Pass --manually-verified only after listening to every manifest pair")
    unverified = [row["sample_id"] for row in rows if row.get("manually_verified", "NO").upper() != "YES"]
    if unverified:
        raise RuntimeError(f"Manifest rows are not manually verified: {unverified[:5]}")
    return rows


def resolve_audio(value: str):
    if "::" in value:
        archive, member = value.split("::", 1)
        return (Path(archive), member)
    return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint containing the 73-token processor")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/mms-300m")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--manually-verified", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence
    from transformers import Wav2Vec2ForCTC, get_linear_schedule_with_warmup

    rows = read_manifest(args.manifest.resolve(), args.manually_verified)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The tiny MMS-300M overfit test requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Run this diagnostic on exactly one GPU, found {torch.cuda.device_count()}")
    device = torch.device("cuda")
    processor = _load_processor(args.checkpoint.resolve(), args.model_name)
    feature = processor.feature_extractor
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        vocab_size=EXPECTED_VOCAB_SIZE,
        pad_token_id=EXPECTED_BLANK_ID,
        ignore_mismatched_sizes=True,
    )
    model.config.ctc_loss_reduction = "mean"
    model.config.ctc_zero_infinity = False
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    model.to(device).train()
    if model.lm_head.out_features != EXPECTED_VOCAB_SIZE:
        raise AssertionError("Tiny-test CTC head is not 73-dimensional")

    head_parameters = list(model.lm_head.parameters())
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("lm_head.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.head_lr, "weight_decay": 0.0},
            {"params": encoder_parameters, "lr": args.encoder_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, args.warmup_steps), max(1, args.max_steps)
    )
    archive_cache = {}
    target_lists = [processor.tokenizer(row["transcript"]).input_ids for row in rows]
    for target in target_lists:
        if not target or EXPECTED_BLANK_ID in target:
            raise ValueError("Tiny manifest contains an empty target or CTC blank ID")

    def make_batch(batch_indices):
        arrays = [torch.from_numpy(read_audio(resolve_audio(rows[index]["audio"]), archive_cache)) for index in batch_indices]
        processed = feature(
            [array.numpy() for array in arrays],
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        labels = [torch.tensor(target_lists[index], dtype=torch.long) for index in batch_indices]
        padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        lengths = torch.tensor([len(target) for target in labels], dtype=torch.long)
        return {
            "input_values": processed["input_values"].to(device),
            "attention_mask": processed["attention_mask"].to(device),
            "input_lengths": processed["attention_mask"].sum(-1).long().to(device),
            "targets": padded.to(device),
            "target_lengths": lengths.to(device),
        }

    def ctc_step(batch):
        outputs = model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits.float()
        assert logits.ndim == 3 and logits.shape[-1] == EXPECTED_VOCAB_SIZE
        output_lengths = feature_output_lengths(model, batch["input_lengths"])
        flat_targets = torch.cat(
            [labels[: int(length.item())] for labels, length in zip(batch["targets"], batch["target_lengths"])]
        )
        required = []
        offset = 0
        for length in batch["target_lengths"].tolist():
            target = flat_targets[offset : offset + int(length)]
            required.append(int(length) + int((target[1:] == target[:-1]).sum().item()))
            offset += int(length)
        required = torch.tensor(required, device=device)
        if (output_lengths < required).any():
            raise RuntimeError(
                f"Invalid CTC lengths: output={output_lengths.tolist()} required={required.tolist()}"
            )
        loss = F.ctc_loss(
            logits.log_softmax(-1).transpose(0, 1),
            flat_targets,
            output_lengths,
            batch["target_lengths"],
            blank=EXPECTED_BLANK_ID,
            zero_infinity=False,
        )
        return loss, logits, output_lengths

    def evaluate():
        model.eval()
        frame_ids = []
        blank_probs = []
        predictions = []
        references = []
        with torch.inference_mode():
            for start in range(0, len(rows), max(1, args.batch_size)):
                indices = list(range(start, min(len(rows), start + args.batch_size)))
                batch = make_batch(indices)
                out = model(input_values=batch["input_values"], attention_mask=batch["attention_mask"])
                logits = out.logits.float()
                lengths = feature_output_lengths(model, batch["input_lengths"])
                ids = logits.argmax(-1)
                probs = logits.softmax(-1)
                for local, index in enumerate(indices):
                    length = int(lengths[local].item())
                    sequence = ids[local, :length].cpu().tolist()
                    frame_ids.extend(sequence)
                    blank_probs.append(float(probs[local, :length, EXPECTED_BLANK_ID].mean().item()))
                    predictions.append(_decode(processor, sequence, EXPECTED_BLANK_ID))
                    references.append(rows[index]["transcript"])
        model.train()
        blank_fraction = sum(token == EXPECTED_BLANK_ID for token in frame_ids) / max(1, len(frame_ids))
        delimiter_fraction = sum(token == EXPECTED_DELIMITER_ID for token in frame_ids) / max(1, len(frame_ids))
        cer_distance = sum(edit_distance(list(ref.replace(" ", "")), list(pred.replace(" ", ""))) for ref, pred in zip(references, predictions))
        wer_distance = sum(edit_distance(ref.split(), pred.split()) for ref, pred in zip(references, predictions))
        return {
            "blank_fraction": blank_fraction,
            "delimiter_fraction": delimiter_fraction,
            "blank_mean_probability": sum(blank_probs) / max(1, len(blank_probs)),
            "empty_prediction_rate": sum(not pred for pred in predictions) / max(1, len(predictions)),
            "cer": cer_distance / max(1, sum(len(ref.replace(" ", "")) for ref in references)),
            "wer": wer_distance / max(1, sum(len(ref.split()) for ref in references)),
            "predictions": [{"reference": ref, "prediction": pred} for ref, pred in zip(references, predictions)],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "tiny_overfit_history.jsonl"
    best = None
    for step in range(1, args.max_steps + 1):
        indices = [((step - 1) * max(1, args.batch_size) + offset) % len(rows) for offset in range(max(1, args.batch_size))]
        batch = make_batch(indices)
        optimizer.zero_grad(set_to_none=True)
        loss, logits, _ = ctc_step(batch)
        if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
            raise FloatingPointError("Tiny overfit produced non-finite loss or logits")
        loss.backward()
        head_norm = grad_norm(head_parameters)
        encoder_norm = grad_norm(encoder_parameters)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % max(1, args.eval_every) == 0 or step == 1:
            metrics = evaluate()
            record = {
                "step": step,
                "loss": float(loss.item()),
                "head_grad_norm": head_norm,
                "encoder_grad_norm": encoder_norm,
                "head_lr": optimizer.param_groups[0]["lr"],
                "encoder_lr": optimizer.param_groups[1]["lr"],
                "loss_finite": True,
                "logits_finite": True,
                **metrics,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if best is None or record["cer"] < best["cer"]:
                best = record
                torch.save(model.state_dict(), args.output_dir / "tiny_overfit_best.pt")

    status = {
        "status": "ok",
        "best": best,
        "passed": bool(best and best["cer"] <= 0.05 and best["wer"] <= 0.05 and best["empty_prediction_rate"] < 0.10),
        "configuration": {
            "samples": len(rows),
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "model": args.model_name,
            "moe": False,
            "dialect_loss": False,
            "augmentation": False,
            "ctc_blank_id": EXPECTED_BLANK_ID,
            "vocabulary_size": EXPECTED_VOCAB_SIZE,
        },
    }
    (args.output_dir / "tiny_overfit_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
