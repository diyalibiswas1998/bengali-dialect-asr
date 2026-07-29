#!/usr/bin/env python
"""Validate processed dataset invariants before expensive training."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import pyarrow.dataset as pads
import soundfile as sf

from asr_dialect_benchmark.data.build_vaani import update_record_fingerprint, wav2vec2_output_frames
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer, normalize_bengali_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--decode-audio", action="store_true", help="Decode every FLAC; recommended before publication")
    args = parser.parse_args()
    root = Path(args.data_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    tokenizer = SimpleTokenizer.load(root / "vocab.json")
    speakers, sample_ids = {}, set()
    counts = {}
    for split in ("train", "validation", "test"):
        assert (root / "manifests" / f"{split}.jsonl").exists(), f"missing {split} manifest"
        dataset = pads.dataset(root / split, format="parquet")
        expected_schema = metadata["schema"]
        assert dataset.schema.names == expected_schema, f"schema mismatch in {split}"
        split_hash = hashlib.sha256()
        count = 0
        for batch in dataset.to_batches(batch_size=512):
            for row in batch.to_pylist():
                count += 1
                assert row["sample_id"] not in sample_ids, f"duplicate sample_id {row['sample_id']}"
                sample_ids.add(row["sample_id"])
                update_record_fingerprint(split_hash, row)
                previous = speakers.setdefault(row["speaker_id"], split)
                assert previous == split, f"speaker overlap: {row['speaker_id']} in {previous} and {split}"
                assert row["transcript"] and row["transcript"] == normalize_bengali_text(row["transcript"])
                assert 0.5 <= row["duration"] <= 30.0
                assert row["dialect_label"] == -100 or 0 <= row["dialect_label"] < 4
                assert row["dialect_label_mask"] == (row["dialect_label"] >= 0)
                if split == "train":
                    assert tokenizer.unk_token_id not in tokenizer.encode_transcript(row["transcript"])
                transcript_ids = tokenizer.encode_transcript(row["transcript"])
                minimum_frames = len(transcript_ids) + sum(
                    left == right for left, right in zip(transcript_ids, transcript_ids[1:])
                )
                assert wav2vec2_output_frames(round(row["duration"] * 16_000)) >= minimum_frames
                if args.decode_audio:
                    audio, sample_rate = sf.read(io.BytesIO(row["audio_flac"]), dtype="float32")
                    assert sample_rate == 16_000 and audio.ndim == 1
                    assert abs(len(audio) / sample_rate - row["duration"]) < 0.02
        counts[split] = count
        assert split_hash.hexdigest() == metadata["splits"][split]["content_sha256"], f"fingerprint mismatch in {split}"
        assert count == metadata["splits"][split]["samples"], f"metadata count mismatch in {split}"
    for artifact in root.rglob("*"):
        if artifact.is_file() and artifact.suffix.lower() in {".json", ".yaml", ".yml", ".md", ".txt"}:
            text = artifact.read_text(encoding="utf-8", errors="ignore")
            assert not re.search(r"hf_[A-Za-z0-9]{20,}", text), f"possible Hugging Face secret in {artifact}"
    assert hashlib.sha256((root / "vocab.json").read_bytes()).hexdigest() == metadata["artifact_hashes"]["vocab_sha256"]
    assert hashlib.sha256((root / "dialect_mapping.json").read_bytes()).hexdigest() == metadata["artifact_hashes"]["dialect_mapping_sha256"]
    print(json.dumps({"valid": True, "counts": counts, "speakers": len(speakers), "vocab_size": len(tokenizer.vocab)}, indent=2))


if __name__ == "__main__":
    main()
