from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from asr_dialect_benchmark.common.constants import DIALECT_TO_IDX, DISTRICT_TO_DIALECT
from asr_dialect_benchmark.data.build_vaani import (
    BuildOptions,
    SCHEMA,
    assign_speaker_splits,
    build_processed_corpus,
    normalize_district,
    update_record_fingerprint,
    wav2vec2_output_frames,
)
from asr_dialect_benchmark.data.processed_vaani import ProcessedVaaniDataset
from asr_dialect_benchmark.data.streaming_vaani import (
    StreamingOptions,
    VaaniStreamingDataset,
    fixed_bengali_tokenizer,
    local_parquets_by_config,
    speaker_split,
)
from asr_dialect_benchmark.training.sampler import LengthBucketBatchSampler
from asr_dialect_benchmark.evaluation.metrics import asr_rates, classification_report
from asr_dialect_benchmark.losses.ctc_losses import multitask_loss
from asr_dialect_benchmark.modeling.asr_model import BengaliDialectASR
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer, normalize_bengali_text


def test_mapping_and_schema_are_versioned_interface():
    assert len(DIALECT_TO_IDX) == 4
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


def test_bengali_normalization_and_training_vocab(tmp_path):
    normalized = normalize_bengali_text("\u200bবাংলা,  test!  কথা")
    assert normalized == "বাংলা কথা"
    tokenizer = SimpleTokenizer()
    tokenizer.fit_from_transcripts([normalized])
    assert tokenizer.decode_ids(tokenizer.encode_transcript(normalized)) == normalized
    tokenizer.save(tmp_path / "vocab.json")
    assert SimpleTokenizer.load(tmp_path / "vocab.json").vocab == tokenizer.vocab


def test_speaker_split_is_deterministic_and_disjoint():
    stats = defaultdict(Counter)
    for speaker in range(100):
        stats[f"speaker-{speaker}"][(speaker % 4, f"district-{speaker % 11}")] = 1 + speaker % 3
    first = assign_speaker_splits(stats, seed=7)
    second = assign_speaker_splits(stats, seed=7)
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    assert len(first) == len(set(first))


def test_metrics_known_values():
    rows = [{"reference": "আমি বাংলা", "prediction": "আমি", "speaker_id": "s1"}]
    scores = asr_rates(rows)
    assert scores["wer"] == 0.5
    report = classification_report([0, 1, 2, 3], [0, 1, 2, 3])
    assert report["macro_f1"] == 1.0


def test_fingerprint_covers_transcript_labels_mapping_inputs():
    record = {
        "sample_id": "id",
        "audio_flac": b"audio",
        "transcript": "কথা",
        "speaker_id": "speaker",
        "source_district": "Kolkata",
        "residence_district": "Kolkata",
        "dialect_group": "Rarhi",
        "dialect_label": 0,
        "dialect_label_mask": True,
    }
    first = hashlib.sha256()
    update_record_fingerprint(first, record)
    second = hashlib.sha256()
    update_record_fingerprint(second, {**record, "transcript": "বাংলা"})
    assert first.hexdigest() != second.hexdigest()


def test_storage_local_sampler_keeps_row_groups_contiguous():
    durations = [4.0, 1.0, 3.0, 2.0, 8.0, 5.0, 7.0, 6.0]
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    sampler = LengthBucketBatchSampler(durations, batch_size=1, seed=3, storage_groups=groups)
    ordered = [batch[0] for batch in sampler]
    transitions = sum(groups[left] != groups[right] for left, right in zip(ordered, ordered[1:]))
    assert transitions == 1
    assert wav2vec2_output_frames(16_000) > 0


def test_synthetic_corpus_builds_and_loads_end_to_end(tmp_path, monkeypatch):
    districts = list(DISTRICT_TO_DIALECT)
    rows = []
    for index in range(12):
        district = districts[index % len(districts)]
        rows.append(
            {
                "audio": {
                    "array": np.full(16_000, (index + 1) / 100.0, dtype=np.float32),
                    "sampling_rate": 16_000,
                },
                "transcript": "বাংলা কথা",
                "language": "Bengali",
                "district": district,
                "speakerID": f"speaker-{index}",
                "duration": 1.0,
                "residence_district": district,
                "sample_id": f"sample-{index}",
            }
        )

    monkeypatch.setattr(
        "asr_dialect_benchmark.data.build_vaani._hf_streams",
        lambda options: iter([("Bengali", rows)]),
    )
    output = build_processed_corpus(
        BuildOptions(output_dir=str(tmp_path / "processed"), shard_size=3)
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert sum(metadata["splits"][split]["samples"] for split in metadata["splits"]) == 12
    assert (output / "vocab.json").exists()
    assert (output / "dialect_mapping.json").exists()

    loaded = 0
    for split in ("train", "validation", "test"):
        dataset = ProcessedVaaniDataset(output, split)
        loaded += len(dataset)
        if len(dataset):
            sample = dataset[0]
            assert sample["input_values"].numel() == 16_000
            assert sample["speaker_id"].startswith("speaker-")
    assert loaded == 12


def test_full_model_and_multitask_loss_forward_backward(monkeypatch):
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
    input_values = torch.randn(2, 160)
    attention_mask = torch.ones_like(input_values, dtype=torch.long)
    outputs = model(
        input_values=input_values,
        attention_mask=attention_mask,
        input_lengths=torch.tensor([160, 144]),
    )
    batch = {
        "targets": torch.tensor([1, 2, 3, 2, 3, 4], dtype=torch.long),
        "target_lengths": torch.tensor([3, 3], dtype=torch.long),
        "dialect_labels": torch.tensor([0, 1], dtype=torch.long),
        "dialect_label_mask": torch.tensor([True, True]),
    }
    loss, components = multitask_loss(outputs, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert model.ctc_head.weight.grad is not None
    assert model.moe.router.proj2.weight.grad is not None
    assert any(parameter.requires_grad for parameter in model.encoder.encoder.layers[-2:].parameters())


def test_direct_stream_preserves_16khz_waveform_and_speaker_split(monkeypatch):
    train_speaker = next(
        f"stream-speaker-{index}"
        for index in range(10_000)
        if speaker_split(f"stream-speaker-{index}", 42) == "train"
    )
    waveform = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    row = {
        "audio": {"array": waveform, "sampling_rate": 16_000},
        "transcript": "বাংলা কথা",
        "language": "Bengali",
        "district": "Kolkata",
        "speakerID": train_speaker,
        "residence_district": "Kolkata",
        "sample_id": "direct-sample",
    }
    tokenizer = fixed_bengali_tokenizer()
    dataset = VaaniStreamingDataset(
        StreamingOptions(
            split="train",
            token="",
            revision="synthetic-revision",
            allow_hf_fallback=False,
            max_samples=1,
        ),
        tokenizer,
    )
    monkeypatch.setattr(dataset, "_source_streams", lambda: iter([("WestBengal_Kolkata", [row])]))
    sample = next(iter(dataset))

    assert torch.equal(sample["input_values"], torch.from_numpy(waveform))
    assert tokenizer.unk_token_id not in sample["target"].tolist()
    assert sample["dialect_label_mask"].item()
    assert sample["source_district"] == "Kolkata"
    assignments = [speaker_split(f"speaker-{index}", 42) for index in range(2_000)]
    assert 0.75 < assignments.count("train") / len(assignments) < 0.85
    assert set(assignments) == {"train", "validation", "test"}


def test_local_only_mode_fails_without_local_files_instead_of_requesting_token(monkeypatch):
    monkeypatch.delenv("VAANI_PARQUET_CACHE", raising=False)
    dataset = VaaniStreamingDataset(
        StreamingOptions(
            split="train",
            token="",
            revision="synthetic-revision",
            allow_hf_fallback=False,
        ),
        fixed_bengali_tokenizer(),
    )
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        next(dataset._source_streams())


def test_paired_audio_loader_uses_only_allowlist_and_supplied_split(tmp_path, monkeypatch):
    split_dir = tmp_path / "train"
    for district in DISTRICT_TO_DIALECT:
        district_dir = split_dir / district
        district_dir.mkdir(parents=True)
        (district_dir / f"{district}_sample.wav").write_bytes(b"synthetic")
        (district_dir / f"{district}_sample.txt").write_text("বাংলা কথা", encoding="utf-8")
    unrelated = split_dir / "Bangalore"
    unrelated.mkdir()
    (unrelated / "ignored.wav").write_bytes(b"synthetic")
    (unrelated / "ignored.txt").write_text("বাংলা", encoding="utf-8")

    dataset = VaaniStreamingDataset(
        StreamingOptions(
            split="train", token="", revision="local", allow_hf_fallback=False
        ),
        fixed_bengali_tokenizer(),
    )
    rows = list(dataset._paired_audio_rows(tmp_path, worker=None))
    assert len(rows) == 11
    assert {row["district"] for row in rows} == set(DISTRICT_TO_DIALECT)
    assert all(row["_preassigned_split"] == "train" for row in rows)
    assert all(row["_dialect_from_district"] for row in rows)

    monkeypatch.setattr(
        "asr_dialect_benchmark.data.streaming_vaani._decode_audio",
        lambda _value: (np.zeros(16_000, dtype=np.float32), 16_000),
    )
    prepared = [dataset._prepare(row, "") for row in rows]
    assert all(sample is not None for sample in prepared)
    assert {sample["dialect_group"] for sample in prepared} == set(DIALECT_TO_IDX)
    assert all(sample["dialect_label_mask"].item() for sample in prepared)


def test_direct_notebook_saves_manifest_when_setup_fails(tmp_path):
    notebook_path = Path(__file__).resolve().parents[1] / "kaggle_direct_vaani_training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    run_dir = tmp_path / "direct-moe-run"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "setup.log").write_text("synthetic secret failure\n", encoding="utf-8")
    namespace = {
        "Path": Path,
        "os": os,
        "shutil": shutil,
        "subprocess": subprocess,
        "time": time,
        "RUN_DIR": run_dir,
        "LOG_DIR": log_dir,
        "REPO_DIR": tmp_path / "missing-repository",
        "SETUP_EXIT_CODE": 1,
        "run_logged": lambda *_args, **_kwargs: 1,
    }

    cells = {cell["id"]: cell for cell in notebook["cells"]}
    for cell_id in ("direct-02", "direct-03", "direct-04", "direct-05"):
        source = "".join(cells[cell_id]["source"])
        exec(compile(source, f"notebook-{cell_id}", "exec"), namespace)
    final_source = "".join(cells["direct-06"]["source"])
    with pytest.raises(RuntimeError, match="Setup failed"):
        exec(compile(final_source, "notebook-direct-06", "exec"), namespace)

    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["setup_exit_code"] == 1
    assert manifest["training_exit_code"] is None
    assert manifest["logs"] == ["setup.log"]


def test_local_parquet_matching_never_assigns_flat_cache_to_kolkata(tmp_path):
    flat = [tmp_path / "train-00000.parquet", tmp_path / "train-00001.parquet"]
    assert local_parquets_by_config(flat) == {}

    district_paths = [
        tmp_path / "Kolkata" / "train-00000.parquet",
        tmp_path / "Dakshin-Dinajpur" / "train-00000.parquet",
    ]
    matches = local_parquets_by_config(district_paths)
    assert matches["WestBengal_Kolkata"] == [district_paths[0]]
    assert matches["WestBengal_DakshinDinajpur"] == [district_paths[1]]


def test_generated_notebook_contains_no_hardcoded_huggingface_token():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "gen_direct_notebook.py").read_text(encoding="utf-8")
    text += (root / "kaggle_direct_vaani_training.ipynb").read_text(encoding="utf-8")
    assert re.search(r"hf_[A-Za-z0-9]{16,}", text) is None


def test_local_four_dialect_config_has_requested_schedule():
    root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(root / "configs" / "local_four_dialect.yaml")
    assert config.model.num_dialects == 4
    assert config.training.steps_per_phase == 1000
    assert config.training.estimated_optimizer_steps_per_phase == 1000
    assert config.training.log_every_steps == 200
    assert config.training.checkpoint_every_steps == 100


def test_local_four_dialect_notebook_is_local_only_and_uses_trainer():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "kaggle_local_four_dialect_training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_text = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"notebook-{cell['id']}", "exec")

    assert 'str(REPO_DIR / "scripts/trainer.py")' in code_text
    assert 'str(REPO_DIR / "configs/local_four_dialect.yaml")' in code_text
    assert "diyalibiswas/vaani-bengali-four-dialect-audio" in code_text
    assert 'Path("/kaggle/input/vaani-bengali-four-dialect-audio")' in code_text
    assert 'os.environ["VAANI_ALLOW_HF_FALLBACK"] = "0"' in code_text
    assert "smoke_direct_streaming.py" not in code_text
    assert "load_dataset(" not in code_text
    assert "--max-train-samples" not in code_text
    assert set(DISTRICT_TO_DIALECT) == {
        "Alipurduar", "CoochBehar", "Darjeeling", "Jalpaiguri",
        "Jhargram", "PaschimMedinipur", "Purulia", "Malda",
        "DakshinDinajpur", "North24Parganas", "Kolkata",
    }
    assert set(DIALECT_TO_IDX) == {"Rarhi", "Varendri", "Jharkhandi", "Kamrupi"}
