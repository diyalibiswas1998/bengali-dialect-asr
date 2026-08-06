from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from asr_dialect_benchmark.common.constants import DIALECT_TO_IDX, DISTRICT_TO_DIALECT
from asr_dialect_benchmark.data.build_vaani import SCHEMA, normalize_district
from asr_dialect_benchmark.data.streaming_vaani import StreamingOptions, VaaniStreamingDataset
from asr_dialect_benchmark.evaluation.extended_metrics import (
    classification_report_imbalanced,
    grouped_asr_report,
)
from asr_dialect_benchmark.losses.ctc_losses import multitask_loss
from asr_dialect_benchmark.modeling.asr_model import BengaliDialectASR
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer, normalize_bengali_text


ROOT = Path(__file__).resolve().parents[1]


def test_mapping_and_schema_contract():
    assert set(DIALECT_TO_IDX) == {"Rarhi", "Varendri", "Jharkhandi", "Kamrupi"}
    assert len(DISTRICT_TO_DIALECT) == 11
    assert DISTRICT_TO_DIALECT["Alipurduar"] == "Kamrupi"
    assert DISTRICT_TO_DIALECT["PaschimMedinipur"] == "Jharkhandi"
    assert normalize_district("North 24 Parganas(20)") == "North24Parganas"
    assert SCHEMA.names == [
        "sample_id", "audio_flac", "duration", "transcript", "speaker_id",
        "source_district", "residence_district", "dialect_group",
        "dialect_label", "dialect_label_mask",
    ]


def test_bengali_normalization_and_tokenizer_round_trip(tmp_path):
    normalized = normalize_bengali_text("\u200bবাংলা, test! কথা")
    assert normalized == "বাংলা কথা"
    tokenizer = SimpleTokenizer()
    tokenizer.fit_from_transcripts([normalized])
    assert tokenizer.decode_ids(tokenizer.encode_transcript(normalized)) == normalized
    tokenizer.save(tmp_path / "vocab.json")
    assert SimpleTokenizer.load(tmp_path / "vocab.json").vocab == tokenizer.vocab


def test_local_pair_loader_ignores_unapproved_district(tmp_path):
    split_dir = tmp_path / "train"
    for district in DISTRICT_TO_DIALECT:
        district_dir = split_dir / district
        district_dir.mkdir(parents=True)
        (district_dir / "sample.wav").write_bytes(b"synthetic")
        (district_dir / "sample.txt").write_text("বাংলা কথা", encoding="utf-8")
    unrelated = split_dir / "Bangalore"
    unrelated.mkdir()
    (unrelated / "ignored.wav").write_bytes(b"synthetic")
    (unrelated / "ignored.txt").write_text("বাংলা", encoding="utf-8")

    dataset = VaaniStreamingDataset(
        StreamingOptions(
            split="train", token="", revision="local", allow_hf_fallback=False
        ),
        SimpleTokenizer(),
    )
    rows = list(dataset._paired_audio_rows(tmp_path, worker=None))
    assert len(rows) == 11
    assert {row["district"] for row in rows} == set(DISTRICT_TO_DIALECT)


def test_model_ctc_and_moe_backward(monkeypatch):
    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.projection = nn.Linear(1, 8)
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(6))
            self.encoder.layer_norm = nn.LayerNorm(8)

        def gradient_checkpointing_enable(self):
            return None

        def _get_feat_extract_output_lengths(self, lengths):
            return torch.div(lengths + 3, 4, rounding_mode="floor")

        def forward(self, input_values, attention_mask=None):
            hidden = self.projection(input_values[:, ::4].unsqueeze(-1))
            for layer in self.encoder.layers:
                hidden = torch.tanh(layer(hidden))
            return SimpleNamespace(last_hidden_state=self.encoder.layer_norm(hidden))

    monkeypatch.setattr(
        "asr_dialect_benchmark.modeling.asr_model.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: FakeEncoder(),
    )
    model = BengaliDialectASR(
        {
            "pretrained_model": "synthetic",
            "num_tokens": 16,
            "num_dialects": 4,
            "top_k": 2,
            "dropout": 0.0,
            "gradient_checkpointing": False,
        }
    )
    model.set_phase(2, top_layers=2)
    outputs = model(
        input_values=torch.randn(2, 160),
        attention_mask=torch.ones(2, 160, dtype=torch.long),
        input_lengths=torch.tensor([160, 144]),
    )
    batch = {
        "targets": torch.tensor([1, 2, 3, 2, 3, 4]),
        "target_lengths": torch.tensor([3, 3]),
        "dialect_labels": torch.tensor([0, 1]),
        "dialect_label_mask": torch.tensor([True, True]),
    }
    loss, components = multitask_loss(outputs, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert model.ctc_head.weight.grad is not None
    assert model.moe.router.proj2.weight.grad is not None


def test_imbalance_metrics_remain_available():
    rows = [
        {
            "reference": "আমি বাংলা",
            "prediction": "আমি বাংলা" if index != 3 else "আমি",
            "dialect_group": ("Rarhi", "Varendri", "Jharkhandi", "Kamrupi")[label],
            "source_district": f"district-{label}",
        }
        for index, label in enumerate([0, 0, 0, 0, 1, 2, 3])
    ]
    y_true = np.asarray([0, 0, 0, 0, 1, 2, 3])
    y_pred = np.asarray([0, 0, 0, 0, 0, 0, 0])
    probabilities = np.full((len(y_true), 4), 0.05)
    probabilities[np.arange(len(y_true)), y_pred] = 0.85
    report = classification_report_imbalanced(
        y_true, y_pred, probabilities, ["Rarhi", "Varendri", "Jharkhandi", "Kamrupi"]
    )
    assert report["imbalance_ratio_max_to_min"] == 4.0
    grouped = grouped_asr_report(rows, "dialect_group")
    assert grouped["overall_micro"]["utterances"] == 7


def test_diagnostic_module_has_audio_finite_dependency():
    path = ROOT / "scripts" / "ctc_collapse_diagnostics.py"
    spec = importlib.util.spec_from_file_location("ctc_collapse_diagnostics_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.np is np


def test_maintained_kaggle_notebook_and_metadata_are_valid():
    directory = ROOT / "kaggle_upload" / "ctc_tiny_overfit"
    metadata = json.loads((directory / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook_path = directory / metadata["code_file"]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == [
        "diyalibiswas/four-dialect-data-undersampled",
        "diyalibiswas/output",
    ]
    assert "CUDA_VISIBLE_DEVICES" in notebook_path.read_text(encoding="utf-8")


def test_readme_records_the_current_blocker():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    incident = (ROOT / "docs" / "CTC_DELIMITER_COLLAPSE.md").read_text(encoding="utf-8")
    assert "delimiter-token collapse" in readme
    assert "Full MoE training is paused" in readme
    assert "fusion.unsqueeze(1)" in incident
    assert "Unfreezing the top 2–4" in incident
