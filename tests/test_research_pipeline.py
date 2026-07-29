from collections import Counter, defaultdict
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

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
    speaker_split,
)
from asr_dialect_benchmark.training.sampler import LengthBucketBatchSampler
from asr_dialect_benchmark.evaluation.metrics import asr_rates, classification_report
from asr_dialect_benchmark.losses.ctc_losses import multitask_loss
from asr_dialect_benchmark.modeling.asr_model import BengaliDialectASR
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer, normalize_bengali_text


def test_mapping_and_schema_are_versioned_interface():
    assert len(DIALECT_TO_IDX) == 4
    assert len(DISTRICT_TO_DIALECT) == 11
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
            token="synthetic-token",
            revision="synthetic-revision",
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
