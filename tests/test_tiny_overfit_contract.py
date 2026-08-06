from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.tiny_overfit_ctc import (
    EXPECTED_BLANK_ID,
    EXPECTED_DELIMITER_ID,
    Sample,
    atomic_json,
    collapse_ctc,
    gate_passed,
    parse_manifest,
)


def test_ctc_collapse_removes_only_blank_and_repeats():
    # The delimiter remains a real token; a blank between equal characters
    # prevents the characters from being collapsed together.
    assert collapse_ctc([0, 2, 2, 0, 3, 0, 3, 0]) == [2, 3, 3]
    assert EXPECTED_BLANK_ID == 0
    assert EXPECTED_DELIMITER_ID == 2


def test_gate_requires_all_metrics_at_same_checkpoint():
    passing = {"cer": 0.04, "wer": 0.05, "empty_prediction_rate": 0.09}
    assert gate_passed(passing)
    assert not gate_passed({**passing, "cer": 0.06})
    assert not gate_passed({**passing, "wer": 0.06})
    assert not gate_passed({**passing, "empty_prediction_rate": 0.10})


def test_atomic_json_writes_valid_utf8(tmp_path):
    target = tmp_path / "status.json"
    atomic_json(target, {"status": "running", "reference": "বাংলা"})
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "running"
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_requires_exactly_32_verified_rows(tmp_path):
    path = tmp_path / "manifest.csv"
    fields = ["sample_id", "audio_path", "transcript", "manually_verified"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(32):
            writer.writerow(
                {
                    "sample_id": f"sample-{index}",
                    "audio_path": f"sample-{index}.wav",
                    "transcript": "বাংলা",
                    "manually_verified": "YES",
                }
            )
    rows = parse_manifest(path, tmp_path, manually_verified=True)
    assert len(rows) == 32
    with path.open("a", encoding="utf-8") as handle:
        handle.write("sample-32,sample-32.wav,বাংলা,YES\n")
    with pytest.raises(ValueError, match="exactly 32"):
        parse_manifest(path, tmp_path, manually_verified=True)


def test_plain_sample_has_no_moe_fields():
    sample = Sample(sample_id="x", audio_path="x.wav", transcript="বাংলা")
    assert not hasattr(sample, "dialect_label")
    assert not hasattr(sample, "router_logits")
