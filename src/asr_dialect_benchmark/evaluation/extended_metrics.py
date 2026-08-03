"""Comprehensive ASR, imbalance-aware classification, and router metrics."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    homogeneity_completeness_v_measure,
    log_loss,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .metrics import edit_distance, error_counts


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / max(1.0, denominator))


def _asr_record(rows) -> dict:
    word_errors, word_total, char_errors, char_total = error_counts(rows)
    return {
        "utterances": len(rows),
        "word_errors": int(word_errors),
        "reference_words": int(word_total),
        "char_errors": int(char_errors),
        "reference_chars": int(char_total),
        "wer": _safe_rate(word_errors, word_total),
        "cer": _safe_rate(char_errors, char_total),
    }


def grouped_asr_report(rows, key: str) -> dict:
    """Return micro, macro, support-weighted, and worst-group ASR metrics."""
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unlabeled")].append(row)
    by_group = {name: _asr_record(group) for name, group in sorted(groups.items())}
    labeled = [(name, score) for name, score in by_group.items() if name != "unlabeled"]
    supports = np.asarray([score["utterances"] for _, score in labeled], dtype=np.float64)
    weights = supports / max(1.0, supports.sum())
    macro = {
        metric: float(np.mean([score[metric] for _, score in labeled])) if labeled else 0.0
        for metric in ("wer", "cer")
    }
    weighted = {
        metric: float(np.sum(weights * [score[metric] for _, score in labeled])) if labeled else 0.0
        for metric in ("wer", "cer")
    }
    worst = {
        metric: max(
            ({"group": name, metric: score[metric]} for name, score in labeled),
            key=lambda item: item[metric],
            default={"group": None, metric: 0.0},
        )
        for metric in ("wer", "cer")
    }
    return {
        "overall_micro": _asr_record(rows),
        "by_group": by_group,
        "macro": macro,
        "support_weighted": weighted,
        "worst_group": worst,
    }


def _expected_calibration_error(y_true, probabilities, bins: int = 15) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = prediction == np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if selected.any():
            value += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
    return float(value)


def classification_report_imbalanced(
    y_true, y_pred, probabilities, class_names
) -> dict:
    """Metrics that remain informative under strong class imbalance."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    specificity = []
    for index in labels:
        true_positive = matrix[index, index]
        false_positive = predicted_support[index] - true_positive
        false_negative = support[index] - true_positive
        true_negative = matrix.sum() - true_positive - false_positive - false_negative
        specificity.append(_safe_rate(true_negative, true_negative + false_positive))
    normalized = np.divide(
        matrix,
        support[:, None],
        out=np.zeros_like(matrix, dtype=np.float64),
        where=support[:, None] != 0,
    )
    nonzero_support = support[support > 0]
    one_hot = label_binarize(y_true, classes=labels)
    probability_metrics = {
        "multiclass_log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error_15_bins": _expected_calibration_error(
            y_true, probabilities, bins=15
        ),
        "top2_accuracy": float(
            np.mean([truth in indices for truth, indices in zip(y_true, np.argsort(probabilities, axis=1)[:, -2:])])
        ),
    }
    try:
        probability_metrics.update(
            roc_auc_ovr_macro=float(
                roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
            ),
            roc_auc_ovr_weighted=float(
                roc_auc_score(one_hot, probabilities, average="weighted", multi_class="ovr")
            ),
            average_precision_macro=float(
                average_precision_score(one_hot, probabilities, average="macro")
            ),
            average_precision_weighted=float(
                average_precision_score(one_hot, probabilities, average="weighted")
            ),
        )
    except ValueError:
        probability_metrics.update(
            roc_auc_ovr_macro=None,
            roc_auc_ovr_weighted=None,
            average_precision_macro=None,
            average_precision_weighted=None,
        )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall_sensitivity": float(recall[index]),
            "specificity": float(specificity[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "predicted_support": int(predicted_support[index]),
        }
        for index, name in enumerate(class_names)
    }
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0
    weighted_precision = float(np.average(precision, weights=support)) if support.sum() else 0.0
    weighted_recall = float(np.average(recall, weights=support)) if support.sum() else 0.0
    weighted_specificity = float(np.average(specificity, weights=support)) if support.sum() else 0.0
    geometric_mean_recall = float(np.prod(recall) ** (1.0 / len(recall))) if np.all(recall > 0) else 0.0
    return {
        "samples": int(len(y_true)),
        "class_support": {name: int(support[index]) for index, name in enumerate(class_names)},
        "imbalance_ratio_max_to_min": float(nonzero_support.max() / nonzero_support.min())
        if len(nonzero_support)
        else 0.0,
        "majority_class_baseline_accuracy": float(nonzero_support.max() / support.sum())
        if support.sum()
        else 0.0,
        "accuracy_micro": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy_macro_recall": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "precision_weighted": weighted_precision,
        "recall_macro": float(np.mean(recall)),
        "recall_weighted": weighted_recall,
        "specificity_macro": float(np.mean(specificity)),
        "specificity_weighted": weighted_specificity,
        "f1_macro": float(np.mean(f1)),
        "f1_micro": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": weighted_f1,
        "geometric_mean_recall": geometric_mean_recall,
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_true_normalized": normalized.tolist(),
        "probability_metrics": probability_metrics,
    }


def router_clustering_report(y_true, gate_probabilities, topk_indices, class_names) -> dict:
    """Evaluate expert routing without assuming expert IDs equal dialect IDs."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(gate_probabilities, dtype=np.float64)
    assignments = probabilities.argmax(axis=1)
    labels = np.arange(probabilities.shape[1])
    topk = np.asarray(topk_indices, dtype=np.int64)
    assignment_counts = np.bincount(assignments, minlength=len(labels))
    topk_counts = np.bincount(topk.reshape(-1), minlength=len(labels))
    assignment_fraction = assignment_counts / max(1, assignment_counts.sum())
    topk_fraction = topk_counts / max(1, topk_counts.sum())
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        y_true, assignments
    )
    contingency = confusion_matrix(y_true, assignments, labels=labels)
    contingency_normalized = np.divide(
        contingency,
        contingency.sum(axis=1, keepdims=True),
        out=np.zeros_like(contingency, dtype=np.float64),
        where=contingency.sum(axis=1, keepdims=True) != 0,
    )
    return {
        "interpretation": (
            "Expert IDs are latent and permutation-invariant; clustering and utilization "
            "metrics are reported instead of dialect classification accuracy."
        ),
        "mean_gate_probability": probabilities.mean(axis=0).tolist(),
        "top1_assignment_fraction": assignment_fraction.tolist(),
        "top2_assignment_fraction": topk_fraction.tolist(),
        "top1_assignment_counts": assignment_counts.tolist(),
        "top2_assignment_counts": topk_counts.tolist(),
        "mean_normalized_routing_entropy": float(entropy.mean() / math.log(len(labels))),
        "top1_utilization_coefficient_of_variation": float(
            assignment_fraction.std() / max(1e-12, assignment_fraction.mean())
        ),
        "load_balancing_statistic": float(
            len(labels) * np.sum(probabilities.mean(axis=0) * topk_fraction)
        ),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(y_true, assignments)
        ),
        "adjusted_mutual_information": float(
            adjusted_mutual_info_score(y_true, assignments)
        ),
        "adjusted_rand_index": float(adjusted_rand_score(y_true, assignments)),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "dialect_by_expert_counts": {
            name: contingency[index].tolist() for index, name in enumerate(class_names)
        },
        "dialect_by_expert_true_normalized": {
            name: contingency_normalized[index].tolist()
            for index, name in enumerate(class_names)
        },
    }


def stratified_bootstrap_intervals(
    rows, y_true, y_pred, iterations: int = 1000, seed: int = 42
) -> dict:
    """Preserve dialect support while bootstrapping ASR and classification metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    per_row_errors = np.asarray(
        [
            (
                edit_distance(row["reference"].split(), row["prediction"].split()),
                len(row["reference"].split()),
                edit_distance(list(row["reference"]), list(row["prediction"])),
                len(row["reference"]),
            )
            for row in rows
        ],
        dtype=np.int64,
    )
    groups = [np.flatnonzero(y_true == label) for label in sorted(set(y_true.tolist()))]
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("wer", "cer", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}
    class_count = int(y_true.max()) + 1
    for _ in range(iterations):
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        totals = per_row_errors[sampled].sum(axis=0)
        values["wer"].append(_safe_rate(totals[0], totals[1]))
        values["cer"].append(_safe_rate(totals[2], totals[3]))
        truth, prediction = y_true[sampled], y_pred[sampled]
        matrix = np.bincount(
            truth * class_count + prediction, minlength=class_count * class_count
        ).reshape(class_count, class_count)
        support = matrix.sum(axis=1)
        predicted_support = matrix.sum(axis=0)
        diagonal = np.diag(matrix)
        recall = np.divide(diagonal, support, out=np.zeros(class_count), where=support != 0)
        precision = np.divide(
            diagonal, predicted_support, out=np.zeros(class_count), where=predicted_support != 0
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(class_count),
            where=(precision + recall) != 0,
        )
        values["accuracy"].append(_safe_rate(diagonal.sum(), matrix.sum()))
        values["balanced_accuracy"].append(float(recall.mean()))
        values["macro_f1"].append(float(f1.mean()))
        values["weighted_f1"].append(float(np.average(f1, weights=support)))

    def summarize(metric_values):
        array = np.asarray(metric_values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "95ci": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        }

    return {
        "method": "utterance bootstrap stratified by dialect label",
        "iterations": int(iterations),
        "seed": int(seed),
        **{name: summarize(metric_values) for name, metric_values in values.items()},
    }
