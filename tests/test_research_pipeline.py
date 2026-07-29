from collections import Counter, defaultdict
import hashlib

from asr_dialect_benchmark.common.constants import DIALECT_TO_IDX, DISTRICT_TO_DIALECT
from asr_dialect_benchmark.data.build_vaani import (
    SCHEMA,
    assign_speaker_splits,
    normalize_district,
    update_record_fingerprint,
    wav2vec2_output_frames,
)
from asr_dialect_benchmark.training.sampler import LengthBucketBatchSampler
from asr_dialect_benchmark.evaluation.metrics import asr_rates, classification_report
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
