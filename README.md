# Bengali Dialect ASR: MMS-300M + Sparse MoE

This repository implements a reproducible comparison between an MMS-300M Bengali CTC baseline and a four-expert dialect-aware MoE. It supports both a validated Parquet derivative and direct paired WAV/TXT folders attached locally on Kaggle.

The dialect labels are **provisional geographic proxies**, not linguistic ground truth. The versioned mapping is in `src/asr_dialect_benchmark/common/constants.py`; Darjeeling and North 24 Parganas are treated as boundary cases in evaluation.

The maintained four-class mapping is: Rarhi (Kolkata, North24Parganas), Varendri (Malda, DakshinDinajpur), Jharkhandi (Jhargram, PaschimMedinipur, Purulia), and Kamrupi (Alipurduar, CoochBehar, Darjeeling, Jalpaiguri).

## 1. Build the processed corpus once

Accept the applicable Hugging Face dataset terms and set `HF_TOKEN` without placing it in a notebook or config. Then run:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/build_processed_vaani.py \
  --source auto \
  --output-dir /path/to/vaani-bengali-processed
python scripts/validate_processed.py \
  --data-dir /path/to/vaani-bengali-processed \
  --decode-audio
```

`auto` first probes the `Bengali` configuration of `ARTPARK-IISc/Vaani-transcription-part`. It uses that source only when audio, transcript, speaker, duration, district, and residence metadata are present. Authentication/network failures stop immediately. If the fields are missing, rerun with `--allow-main-fallback` to explicitly acknowledge filtering the much larger 11-district raw corpus.

Records are NFC-normalized Bengali transcripts with mono 16 kHz FLAC bytes. Rows must have a speaker ID, decode successfully, be 0.5–30 seconds, be CTC-alignable, and be unique by stable ID and audio hash. Only parsed residence districts receive a dialect label; other valid rows remain CTC examples with label `-100`. Speakers are assigned globally to deterministic 80/10/10 splits. Failed builds preserve validated staging; resume them with `--resume-staging /path/to/.building-directory`.

The output contains `train/`, `validation/`, and `test/` Parquet shards, JSONL split manifests, `metadata.json`, `vocab.json`, `dialect_mapping.json`, and source/license notes. Upload this directory as a **private Kaggle Dataset**, retaining upstream attribution and terms.

For a quick pipeline check, add `--max-samples 200`; do not use that cap for reported experiments.

## 2. Train on Kaggle T4×2

Attach the processed private dataset locally, then launch two processes:

```bash
accelerate launch --config_file configs/accelerate_t4x2.yaml \
  scripts/train_research.py \
  --data-dir /kaggle/input/vaani-bengali-processed \
  --output-dir /kaggle/working/moe-run \
  --experiment moe \
  --resume latest
```

To publish either maintained notebook through the Kaggle API, configure the Kaggle CLI and run `python scripts/publish_kaggle_notebook.py --username YOUR_USER --notebook creator`; after the processed Dataset exists, publish training with `--notebook training --processed-dataset OWNER/SLUG`. Kernel pushes are private and start a Kaggle execution.

To train directly from the uploaded paired files, use `kaggle_direct_vaani_training.ipynb` or publish it with `python scripts/publish_kaggle_notebook.py --username YOUR_USER --notebook direct`. The attached Dataset must contain `train/`, `validation/`, and `test/`, each with the same 11 district folders and paired `.wav`/`.txt` files. The loader ignores every other district, preserves the supplied splits, globally shuffles the selected training rows, and derives the four dialect proxy labels from the district folder. It uses local-only mode by default and needs no Vaani Hugging Face token.

For the fixed short schedule, use `kaggle_local_four_dialect_training.ipynb` or publish it with `python scripts/publish_kaggle_notebook.py --username YOUR_USER --notebook local-four`. This separate notebook invokes `scripts/trainer.py`, has no preliminary test run, and hard-disables dataset fallback. `configs/local_four_dialect.yaml` runs exactly 1,000 optimizer steps in each of three phases, prints progress every 200 phase steps, and writes resumable checkpoints every 100 global optimizer steps.

Omit `--resume` for the first session. Preserve `/kaggle/working/moe-run` as a private Kaggle Dataset version between sessions, restore it before the next run, and then use `--resume latest`.

The three passes are fixed by `configs/research.yaml`:

1. freeze MMS-300M; train heads, router, four dialect experts, and the shared expert;
2. unfreeze the top four MMS transformer blocks at `1e-5`;
3. continue with those blocks at `5e-6`.

The head/MoE LR is `2e-4`; CTC/dialect/load-balancing weights are `1.0/0.2/0.01`. Dialect supervision aligns both the classifier and router, sparse dispatch evaluates only selected dialect experts, and balancing statistics are recomputed over the gathered effective batch. T4×2 uses FP16, per-device batch 1, accumulation 16 (effective batch 32), storage-local duration buckets, 5% warmup, linear decay, and gradient clipping 1.0. Checkpoints are written every 2,000 optimizer updates and at every phase boundary. They contain model, optimizer, scheduler, scaler, RNG, next batch position, sanitized config, vocabulary, mapping, and content-complete split fingerprints.

Run the identical-split experiments with only `--experiment` changed:

```text
baseline   MMS-300M CTC without MoE
moe        top-2 MoE + dialect + shared expert
top1       top-1 routing ablation
no_dialect no dialect-loss ablation
no_shared  no shared-expert ablation
```

## 3. Validate and evaluate

Before full runs, verify DDP forward/backward and state restoration on T4×2:

```bash
accelerate launch --config_file configs/accelerate_t4x2.yaml \
  scripts/smoke_test_research.py \
  --data-dir /kaggle/input/vaani-bengali-processed \
  --require-two-gpus
```

Evaluate a completed checkpoint:

```bash
python scripts/validate_checkpoint.py \
  --checkpoint /path/to/checkpoint-phase-3 \
  --expected-processes 2
accelerate launch --config_file configs/accelerate_t4x2.yaml \
  scripts/evaluate_research.py \
  --checkpoint /path/to/checkpoint-phase-3 \
  --data-dir /kaggle/input/vaani-bengali-processed
```

The JSON report includes overall WER/CER, macro and per-dialect scores, source- and residence-district scores, dialect-head and router macro-F1/confusion matrices, expert utilization, speaker-bootstrap 95% confidence intervals, and residence-based sensitivity excluding Darjeeling and North 24 Parganas. Router classification is intentionally omitted for the no-dialect-loss ablation because expert IDs are then permutation-invariant.

## Research constraints

- All reported systems must have identical split fingerprints in checkpoint `config.json`.
- “All data” means all valid transcribed Bengali rows, not untranscribed rows.
- Obtain Bengali linguistic review before presenting the mapping as definitive.
- Review the current Vaani and MMS-300M licenses. MMS-300M is intended here for noncommercial research.
- A baseline win or a null result is valid; reproducibility and defensible labels are the success criteria.
