#!/usr/bin/env python
"""Evaluate a trained checkpoint directly from attached WAV/TXT split data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from asr_dialect_benchmark.common.constants import BOUNDARY_DISTRICTS, DIALECT_GROUPS
from asr_dialect_benchmark.data import StreamingOptions, VaaniStreamingDataset, processed_collate
from asr_dialect_benchmark.evaluation.extended_metrics import (
    classification_report_imbalanced,
    grouped_asr_report,
    router_clustering_report,
    stratified_bootstrap_intervals,
)
from asr_dialect_benchmark.modeling import BengaliDialectASR
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer


TENSOR_KEYS = (
    "input_values",
    "attention_mask",
    "input_lengths",
    "targets",
    "target_lengths",
    "dialect_labels",
    "dialect_label_mask",
)


def load_checkpoint_state(checkpoint: Path):
    portable = checkpoint / "model_state.pt"
    safe = checkpoint / "model.safetensors"
    if portable.is_file():
        return torch.load(portable, map_location="cpu", weights_only=True), portable.name
    if safe.is_file():
        return load_safetensors(str(safe), device="cpu"), safe.name
    raise FileNotFoundError(f"No model_state.pt or model.safetensors in {checkpoint}")


def write_rows_csv(path: Path, rows: list[dict], class_names: list[str]) -> None:
    fields = [
        "sample_id", "speaker_id", "source_district", "dialect_group",
        "dialect_label", "dialect_prediction", "router_top1", "duration_seconds",
        "reference", "prediction",
    ]
    fields += [f"dialect_probability_{name}" for name in class_names]
    fields += [f"router_probability_expert_{index}" for index in range(len(class_names))]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in fields}
            for index, name in enumerate(class_names):
                output[f"dialect_probability_{name}"] = row["dialect_probabilities"][index]
                output[f"router_probability_expert_{index}"] = row["router_probabilities"][index]
            writer.writerow(output)


def write_group_csv(path: Path, report: dict, group_name: str) -> None:
    fields = [
        group_name, "utterances", "word_errors", "reference_words", "wer",
        "char_errors", "reference_chars", "cer",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, metrics in report["by_group"].items():
            writer.writerow({group_name: name, **metrics})


def classification_by_district(rows: list[dict]) -> dict:
    grouped = {}
    for district in sorted({row["source_district"] for row in rows}):
        selected = [row for row in rows if row["source_district"] == district]
        correct = sum(row["dialect_label"] == row["dialect_prediction"] for row in selected)
        grouped[district] = {
            "dialect_group": selected[0]["dialect_group"],
            "support": len(selected),
            "accuracy": float(correct / max(1, len(selected))),
            "mean_true_class_probability": float(
                np.mean(
                    [row["dialect_probabilities"][row["dialect_label"]] for row in selected]
                )
            ),
        }
    return grouped


def write_district_classification_csv(path: Path, report: dict) -> None:
    fields = ["district", "dialect_group", "support", "accuracy", "mean_true_class_probability"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for district, metrics in report.items():
            writer.writerow({"district": district, **metrics})


def plot_confusion(report: dict, class_names: list[str], path: Path, normalized: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = "confusion_matrix_true_normalized" if normalized else "confusion_matrix"
    matrix = np.asarray(report[key])
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted dialect")
    axis.set_ylabel("True dialect")
    axis.set_title("Dialect confusion matrix" + (" (true-normalized)" if normalized else ""))
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = f"{value:.2f}" if normalized else str(int(value))
            axis.text(
                column, row, label, ha="center", va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_class_support(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["class_support"])
    values = [report["class_support"][name] for name in names]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(names, values, color="#4C78A8")
    axis.bar_label(bars)
    axis.set_ylabel("Test utterances")
    axis.set_title("Dialect support (class imbalance)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_group_asr(report: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["by_group"])
    cer = [report["by_group"][name]["cer"] for name in names]
    wer = [report["by_group"][name]["wer"] for name in names]
    positions = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    axis.bar(positions - width / 2, cer, width, label="CER")
    axis.bar(positions + width / 2, wer, width, label="WER")
    axis.set_xticks(positions, names, rotation=35, ha="right")
    axis.set_ylabel("Error rate (lower is better)")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_probability_curves(
    y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    binary = label_binarize(y_true, classes=np.arange(len(class_names)))
    roc_figure, roc_axis = plt.subplots(figsize=(7, 6))
    pr_figure, pr_axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(class_names):
        false_positive, true_positive, _ = roc_curve(binary[:, index], probabilities[:, index])
        precision, recall, _ = precision_recall_curve(binary[:, index], probabilities[:, index])
        roc_axis.plot(false_positive, true_positive, label=name)
        pr_axis.plot(recall, precision, label=name)
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    roc_axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC curves")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision-recall curves")
    roc_axis.legend()
    pr_axis.legend()
    roc_figure.tight_layout()
    pr_figure.tight_layout()
    roc_figure.savefig(output_dir / "dialect_roc_curves.png", dpi=180)
    pr_figure.savefig(output_dir / "dialect_precision_recall_curves.png", dpi=180)
    plt.close(roc_figure)
    plt.close(pr_figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional developer-only cap")
    parser.add_argument("--require-two-gpus", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16")
    if args.require_two_gpus and (
        accelerator.num_processes != 2 or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"Expected two GPU processes, got processes={accelerator.num_processes}, "
            f"cuda={torch.cuda.is_available()}"
        )

    config = OmegaConf.create(
        json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    )
    config.model.gradient_checkpointing = False
    tokenizer = SimpleTokenizer.load(checkpoint / "vocab.json")
    model = BengaliDialectASR(config)
    state, state_file = load_checkpoint_state(checkpoint)
    model.load_state_dict(state, strict=True)
    del state
    model.to(accelerator.device)
    model.eval()

    token = os.environ.get("HF_TOKEN", "")
    dataset = VaaniStreamingDataset(
        StreamingOptions(
            split=args.split,
            token=token,
            revision=str(config.data.revision),
            allow_hf_fallback=False,
            seed=int(config.seed),
            epoch=0,
            min_duration=float(config.data.min_duration),
            max_duration=float(config.data.max_duration),
            shuffle_buffer=int(config.data.shuffle_buffer),
            max_samples=args.max_samples,
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
        ),
        tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=processed_collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    local_rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for local_index, batch in enumerate(loader, 1):
            device_batch = {
                key: batch[key].to(accelerator.device, non_blocking=True) for key in TENSOR_KEYS
            }
            with accelerator.autocast():
                outputs = model(
                    device_batch["input_values"],
                    device_batch["attention_mask"],
                    device_batch["input_lengths"],
                )
            output_length = int(outputs["output_lengths"][0].item())
            token_ids = outputs["logits"].argmax(-1)[0, :output_length].tolist()
            dialect_probabilities = outputs["dialect_logits"].softmax(-1)[0].float().cpu().tolist()
            router_probabilities = outputs["gate_probs"][0].float().cpu().tolist()
            topk_indices = outputs["topk_indices"][0].cpu().tolist()
            local_rows.append(
                {
                    "sample_id": batch["sample_id"][0],
                    "speaker_id": batch["speaker_id"][0],
                    "source_district": batch["source_district"][0],
                    "dialect_group": batch["dialect_group"][0],
                    "dialect_label": int(device_batch["dialect_labels"][0].item()),
                    "dialect_prediction": int(np.argmax(dialect_probabilities)),
                    "dialect_probabilities": dialect_probabilities,
                    "router_top1": int(np.argmax(router_probabilities)),
                    "router_probabilities": router_probabilities,
                    "router_topk": topk_indices,
                    "duration_seconds": float(device_batch["input_lengths"][0].item() / 16_000),
                    "reference": batch["transcript"][0],
                    "prediction": tokenizer.decode_ids(token_ids, ctc=True),
                }
            )
            if local_index % args.progress_every == 0:
                print(
                    f"evaluation_progress rank={accelerator.process_index} "
                    f"local_samples={local_index} elapsed_minutes={(time.monotonic()-started)/60:.1f}",
                    flush=True,
                )

    gathered = gather_object(local_rows)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return
    rows = gathered
    if rows and isinstance(rows[0], list):
        rows = [row for process_rows in rows for row in process_rows]
    rows = sorted(rows, key=lambda row: row["sample_id"])
    if not rows:
        raise RuntimeError("No valid evaluation samples were produced")

    class_names = list(DIALECT_GROUPS)
    y_true = np.asarray([row["dialect_label"] for row in rows], dtype=np.int64)
    y_pred = np.asarray([row["dialect_prediction"] for row in rows], dtype=np.int64)
    dialect_probabilities = np.asarray(
        [row["dialect_probabilities"] for row in rows], dtype=np.float64
    )
    router_probabilities = np.asarray(
        [row["router_probabilities"] for row in rows], dtype=np.float64
    )
    router_topk = np.asarray([row["router_topk"] for row in rows], dtype=np.int64)

    asr_by_dialect = grouped_asr_report(rows, "dialect_group")
    asr_by_district = grouped_asr_report(rows, "source_district")
    dialect_report = classification_report_imbalanced(
        y_true, y_pred, dialect_probabilities, class_names
    )
    district_classification = classification_by_district(rows)
    router_report = router_clustering_report(
        y_true, router_probabilities, router_topk, class_names
    )
    boundary_rows = [
        row for row in rows if row["source_district"] not in BOUNDARY_DISTRICTS
    ]
    boundary_true = np.asarray([row["dialect_label"] for row in boundary_rows], dtype=np.int64)
    boundary_pred = np.asarray([row["dialect_prediction"] for row in boundary_rows], dtype=np.int64)
    boundary_probabilities = np.asarray(
        [row["dialect_probabilities"] for row in boundary_rows], dtype=np.float64
    )
    bootstrap = stratified_bootstrap_intervals(
        rows, y_true, y_pred, iterations=args.bootstrap_iterations, seed=int(config.seed)
    )
    checkpoint_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_state_file": state_file,
        "checkpoint_training_state": checkpoint_state,
        "split": args.split,
        "evaluated_samples": len(rows),
        "elapsed_minutes": (time.monotonic() - started) / 60,
        "class_order": class_names,
        "asr_by_dialect": asr_by_dialect,
        "asr_by_district": asr_by_district,
        "dialect_head_imbalance_aware": dialect_report,
        "dialect_head_by_district": district_classification,
        "router_permutation_invariant": router_report,
        "mapping_sensitivity_excluding_boundary_districts": {
            "excluded_districts": list(BOUNDARY_DISTRICTS),
            "evaluated_samples": len(boundary_rows),
            "asr_by_dialect": grouped_asr_report(boundary_rows, "dialect_group"),
            "dialect_head_imbalance_aware": classification_report_imbalanced(
                boundary_true, boundary_pred, boundary_probabilities, class_names
            ),
        },
        "stratified_bootstrap_95ci": bootstrap,
        "data_limit": args.max_samples,
    }
    (output_dir / f"evaluation_{args.split}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_rows_csv(output_dir / f"predictions_{args.split}.csv", rows, class_names)
    write_group_csv(output_dir / "asr_by_dialect.csv", asr_by_dialect, "dialect")
    write_group_csv(output_dir / "asr_by_district.csv", asr_by_district, "district")
    write_district_classification_csv(
        output_dir / "dialect_classification_by_district.csv", district_classification
    )
    plot_confusion(dialect_report, class_names, output_dir / "dialect_confusion_raw.png", False)
    plot_confusion(
        dialect_report, class_names, output_dir / "dialect_confusion_normalized.png", True
    )
    plot_class_support(dialect_report, output_dir / "dialect_class_support.png")
    plot_group_asr(asr_by_dialect, output_dir / "asr_by_dialect.png", "ASR error rates by dialect")
    plot_group_asr(asr_by_district, output_dir / "asr_by_district.png", "ASR error rates by district")
    plot_probability_curves(y_true, dialect_probabilities, class_names, output_dir)
    summary = {
        "checkpoint": str(checkpoint),
        "validation_loss_used_for_selection": checkpoint_state.get("validation_loss"),
        "test_samples": len(rows),
        "overall_wer": asr_by_dialect["overall_micro"]["wer"],
        "overall_cer": asr_by_dialect["overall_micro"]["cer"],
        "macro_dialect_wer": asr_by_dialect["macro"]["wer"],
        "macro_dialect_cer": asr_by_dialect["macro"]["cer"],
        "dialect_accuracy": dialect_report["accuracy_micro"],
        "dialect_balanced_accuracy": dialect_report["balanced_accuracy_macro_recall"],
        "dialect_macro_f1": dialect_report["f1_macro"],
        "dialect_weighted_f1": dialect_report["f1_weighted"],
        "dialect_mcc": dialect_report["matthews_correlation_coefficient"],
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
