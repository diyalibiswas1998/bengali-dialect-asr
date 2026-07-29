"""Dependency-light research metrics and speaker bootstrap intervals."""

from __future__ import annotations

import random
from collections import Counter, defaultdict


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, 1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (ref_item != hyp_item)))
        previous = current
    return previous[-1]


def error_counts(rows):
    word_errors = word_total = char_errors = char_total = 0
    for row in rows:
        reference, prediction = row["reference"].strip(), row["prediction"].strip()
        reference_words, prediction_words = reference.split(), prediction.split()
        word_errors += edit_distance(reference_words, prediction_words)
        word_total += len(reference_words)
        char_errors += edit_distance(list(reference), list(prediction))
        char_total += len(reference)
    return word_errors, word_total, char_errors, char_total


def asr_rates(rows):
    word_errors, word_total, char_errors, char_total = error_counts(rows)
    return {"wer": word_errors / max(1, word_total), "cer": char_errors / max(1, char_total), "utterances": len(rows)}


def grouped_asr(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key] or "unlabeled"].append(row)
    scores = {name: asr_rates(group) for name, group in sorted(groups.items())}
    labeled = [score for name, score in scores.items() if name != "unlabeled"]
    macro = {
        "wer": sum(score["wer"] for score in labeled) / max(1, len(labeled)),
        "cer": sum(score["cer"] for score in labeled) / max(1, len(labeled)),
    }
    return scores, macro


def classification_report(true_labels, predicted_labels, num_classes=4):
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, prediction in zip(true_labels, predicted_labels):
        confusion[int(truth)][int(prediction)] += 1
    f1_scores = []
    for label in range(num_classes):
        true_positive = confusion[label][label]
        false_positive = sum(confusion[row][label] for row in range(num_classes) if row != label)
        false_negative = sum(confusion[label][column] for column in range(num_classes) if column != label)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1_scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return {"macro_f1": sum(f1_scores) / num_classes, "per_class_f1": f1_scores, "confusion_matrix": confusion}


def speaker_bootstrap(rows, iterations=1000, seed=42):
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)
    speakers = sorted(by_speaker)
    counts = {speaker: error_counts(items) for speaker, items in by_speaker.items()}
    rng, wer_values, cer_values = random.Random(seed), [], []
    for _ in range(iterations):
        totals = [0, 0, 0, 0]
        for _ in speakers:
            sample = counts[rng.choice(speakers)]
            totals = [left + right for left, right in zip(totals, sample)]
        wer_values.append(totals[0] / max(1, totals[1]))
        cer_values.append(totals[2] / max(1, totals[3]))
    def interval(values):
        ordered = sorted(values)
        return [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]
    return {"iterations": iterations, "speakers": len(speakers), "wer_95ci": interval(wer_values), "cer_95ci": interval(cer_values)}
