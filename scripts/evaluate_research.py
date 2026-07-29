#!/usr/bin/env python
"""Evaluate overall/group/district ASR and MoE routing behavior."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import gather_object
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from asr_dialect_benchmark.common.constants import BOUNDARY_DISTRICTS
from asr_dialect_benchmark.data import ProcessedVaaniDataset, processed_collate
from asr_dialect_benchmark.evaluation.metrics import asr_rates, classification_report, grouped_asr, speaker_bootstrap
from asr_dialect_benchmark.modeling import BengaliDialectASR
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision="fp16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    checkpoint = Path(args.checkpoint)
    config = OmegaConf.create(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    dataset = ProcessedVaaniDataset(args.data_dir, args.split, SimpleTokenizer.load(checkpoint / "vocab.json"))
    model = BengaliDialectASR(config)
    model.load_state_dict(torch.load(checkpoint / "model_state.pt", map_location="cpu", weights_only=True))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=processed_collate, num_workers=2)
    model, loader = accelerator.prepare(model, loader)
    model.eval()
    local_rows = []
    gate_sum = torch.zeros(4, device=accelerator.device)
    gate_count = torch.zeros((), device=accelerator.device)
    utilization = torch.zeros(4, device=accelerator.device)
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["input_values"], batch["attention_mask"], batch["input_lengths"])
            token_ids = outputs["logits"].argmax(-1)[0, : outputs["output_lengths"][0]].tolist()
            row = {
                "prediction": dataset.tokenizer.decode_ids(token_ids, ctc=True),
                "reference": batch["transcript"][0],
                "speaker_id": batch["speaker_id"][0],
                "source_district": batch["source_district"][0],
                "residence_district": batch["residence_district"][0],
                "dialect_group": batch["dialect_group"][0],
                "dialect_label": int(batch["dialect_labels"][0].item()),
                "dialect_head_prediction": int(outputs["dialect_logits"].argmax(-1)[0].item()),
                "router_prediction": None,
            }
            if outputs["gate_probs"] is not None:
                row["router_prediction"] = int(outputs["gate_probs"].argmax(-1)[0].item())
                gate_sum += outputs["gate_probs"][0]
                gate_count += 1
                utilization.scatter_add_(0, outputs["topk_indices"][0], torch.ones_like(outputs["topk_indices"][0], dtype=utilization.dtype))
            local_rows.append(row)
    rows = gather_object(local_rows)
    gate_sum = accelerator.reduce(gate_sum, reduction="sum")
    gate_count = accelerator.reduce(gate_count, reduction="sum")
    utilization = accelerator.reduce(utilization, reduction="sum")
    if not accelerator.is_main_process:
        return

    by_group, group_macro = grouped_asr(rows, "dialect_group")
    by_district, district_macro = grouped_asr(rows, "source_district")
    by_residence_district, residence_district_macro = grouped_asr(rows, "residence_district")
    labeled = [row for row in rows if row["dialect_label"] >= 0]
    router_supervised = float(config.loss.dialect_weight) > 0
    boundary_rows = [row for row in rows if row["residence_district"] not in BOUNDARY_DISTRICTS]
    boundary_by_group, boundary_macro = grouped_asr(boundary_rows, "dialect_group")
    boundary_labeled = [row for row in boundary_rows if row["dialect_label"] >= 0]
    boundary_router = [row for row in boundary_labeled if row["router_prediction"] is not None]
    boundary_report = {
        "excluded_districts": list(BOUNDARY_DISTRICTS),
        "overall": asr_rates(boundary_rows),
        "by_dialect": boundary_by_group,
        "macro_by_dialect": boundary_macro,
        "dialect_head": classification_report(
            [row["dialect_label"] for row in boundary_labeled],
            [row["dialect_head_prediction"] for row in boundary_labeled],
        ) if router_supervised else None,
        "router": classification_report(
            [row["dialect_label"] for row in boundary_router],
            [row["router_prediction"] for row in boundary_router],
        ) if boundary_router and router_supervised else None,
    }
    report = {
        "checkpoint": str(checkpoint),
        "split": args.split,
        "overall": asr_rates(rows),
        "by_dialect": by_group,
        "macro_by_dialect": group_macro,
        "by_district": by_district,
        "macro_by_district": district_macro,
        "by_residence_district": by_residence_district,
        "macro_by_residence_district": residence_district_macro,
        "dialect_head": classification_report([row["dialect_label"] for row in labeled], [row["dialect_head_prediction"] for row in labeled]) if router_supervised else None,
        "mapping_sensitivity_excluding_boundary_districts": boundary_report,
        "speaker_bootstrap": speaker_bootstrap(rows, args.bootstrap_iterations),
    }
    router_rows = [row for row in labeled if row["router_prediction"] is not None]
    if router_rows:
        report["router"] = classification_report([row["dialect_label"] for row in router_rows], [row["router_prediction"] for row in router_rows]) if router_supervised else {
            "classification": None,
            "reason": "Router expert IDs are permutation-invariant when dialect loss is disabled.",
        }
        report["router"]["mean_gate_probability"] = (gate_sum / gate_count.clamp_min(1)).cpu().tolist()
        report["router"]["expert_assignment_fraction"] = (utilization / utilization.sum().clamp_min(1)).cpu().tolist()
        report["router"]["load_balancing_statistic"] = float(
            4 * torch.sum((gate_sum / gate_count.clamp_min(1)) * (utilization / utilization.sum().clamp_min(1))).item()
        )
    else:
        report["router"] = None
    target = Path(args.output) if args.output else checkpoint / f"evaluation_{args.split}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
