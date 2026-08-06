# Bengali Dialect ASR

Research code for Bengali automatic speech recognition across four provisional
dialect groups using `facebook/mms-300m`, CTC, and an experimental sparse
mixture-of-experts (MoE) model.

## Current status — 6 August 2026

**Full MoE training is paused. Do not use the existing phase-3 checkpoint as a
working ASR model.** Its tokenizer, labels, audio preprocessing, checkpoint
weights, and CTC lengths passed audit, but its frame predictions collapsed to
the word-delimiter token.

The audited phase-3 checkpoint produced the following results on 100 validation
utterances:

| Metric | Result |
|---|---:|
| WER | 100.00% |
| CER | 91.87% |
| Delimiter argmax frames | 95.88% |
| Blank argmax frames | 0.00% |
| Empty decoded predictions | 4.00% |
| Invalid CTC lengths | 0 |
| Unknown or blank IDs in valid targets | 0 |

This is **delimiter-token collapse**, not CTC blank collapse. The next required
gate is a manually verified 32-sample, plain MMS-CTC overfit test. The latest
recorded Kaggle check had not yet produced a passing result. Until
`tiny_overfit_status.json` proves otherwise, the repository makes no claim that
the full ASR or MoE training schedule works.

See [docs/CTC_DELIMITER_COLLAPSE.md](docs/CTC_DELIMITER_COLLAPSE.md) for the
evidence, architecture risks, ruled-out causes, and experiment order.

## Maintained entry points

| Purpose | Entry point |
|---|---|
| Kaggle checkpoint audit and 32-sample test | `kaggle_upload/ctc_tiny_overfit/` |
| Checkpoint/token/length audit | `scripts/ctc_collapse_diagnostics.py` |
| Manifest-aware Kaggle wrapper | `scripts/kaggle_ctc_collapse_diagnostics.py` |
| Plain MMS-CTC tiny overfit | `scripts/tiny_overfit_ctc.py` |
| Build an undersampled copy without changing source files | `scripts/build_four_dialect_undersampled.py` |
| Build/validate the processed Parquet corpus | `scripts/build_processed_vaani.py`, `scripts/validate_processed.py` |
| Experimental full research trainer | `scripts/train_research.py` |

Older generated notebooks, compatibility wrappers, account-specific notebook
versions, and one-off patch scripts have been removed. Git is the source of
truth; do not copy code back from failed Kaggle outputs.

## Data contract

The current Kaggle workflow expects either split directories or split ZIP files:

```text
dataset-root/
  train/
    Alipurduar/*.wav + *.txt
    ...
  validation/
    ...
  test/
    ...
```

Every audio file must have a transcript with the same relative stem. Only these
11 district folders are accepted:

| Dialect proxy | Districts |
|---|---|
| Kamrupi | Alipurduar, CoochBehar, Darjeeling, Jalpaiguri |
| Jharkhandi | Jhargram, PaschimMedinipur, Purulia |
| Varendri | Malda, DakshinDinajpur |
| Rarhi | North24Parganas, Kolkata |

These labels are geographic proxies, not definitive linguistic labels.
Darjeeling and North 24 Parganas remain boundary cases.

## Run on Kaggle

The maintained notebook is self-contained and does not stream the speech
dataset. Attach these inputs:

1. `diyalibiswas/four-dialect-data-undersampled`
2. `diyalibiswas/output` (the saved checkpoint and processor)

Enable a GPU and Internet, then run
`kaggle_upload/ctc_tiny_overfit/bengali_ctc_final_confirmed_tiny_overfit.ipynb`.
Internet is needed only for the public MMS-300M model download. No Hugging Face
token is needed for either attached Kaggle dataset.

The notebook:

1. locates the attached data and phase checkpoint;
2. creates a deterministic 32-row manifest;
3. records the user's completed audio/transcript verification;
4. audits the saved checkpoint on 100 validation rows;
5. launches a fresh, one-GPU, plain MMS-CTC overfit test for 3,000 steps; and
6. saves the manifest, audit, history, predictions, and final status as a ZIP.

The test passes only when its best result satisfies all three conditions:

```text
CER <= 0.05
WER <= 0.05
empty_prediction_rate < 0.10
```

Do not start another full MoE run if `tiny_overfit_status.json` reports
`"passed": false`.

### Publish from another Kaggle account

Authenticate the Kaggle CLI for the account you intend to use, then run:

```bash
python scripts/publish_kaggle_diagnostics.py \
  --username YOUR_KAGGLE_USERNAME \
  --audio-dataset OWNER/four-dialect-data-undersampled \
  --checkpoint-dataset OWNER/output
```

The publisher works in a temporary directory, changes only the kernel metadata,
keeps the notebook private, attaches both datasets, and starts a GPU run.

## Run locally

Install the package in an isolated environment:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Audit a checkpoint without training:

```bash
python scripts/ctc_collapse_diagnostics.py \
  --data-root /path/to/four-dialect-data \
  --repo-root . \
  --checkpoint /path/to/checkpoint-phase-3 \
  --output-dir outputs/ctc-audit \
  --sample-count 100 \
  --batch-size 4
```

Create the tiny manifest:

```bash
python scripts/kaggle_ctc_collapse_diagnostics.py \
  --data-root /path/to/four-dialect-data \
  --repo-root . \
  --output-dir outputs/ctc-audit \
  --make-manifest outputs/ctc-audit/tiny_manifest.csv \
  --manifest-count 32
```
Before running, attach the existing validated Bengali processor directory. It must
contain the processor configuration and satisfy vocabulary size 73, blank/pad ID
0, unknown ID 1, delimiter token `|` with ID 2, 16-kHz sampling, and enabled
feature-extractor normalization. The experiment has no `--checkpoint` option:
it never loads or resumes the failed MoE checkpoint.

Listen to every selected recording, compare it with its transcript, and set
`manually_verified` to `YES` only for checked pairs. Then run the fresh plain
MMS-CTC test on one CUDA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tiny_overfit_ctc.py \
  --manifest outputs/ctc-audit/tiny_manifest.csv \
  --processor-path /path/to/validated_bengali_processor \
  --output-dir outputs/tiny-overfit \
  --trainable-encoder-layers 4 \
  --head-lr 1e-3 --encoder-lr 1e-5 \
  --batch-size 1 --gradient-accumulation-steps 8 \
  --max-steps 3000 --eval-steps 25 \
  --seed 42 --fp16 --gradient-checkpointing \
  --manually-verified
```

PowerShell users should set the environment variable separately:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python scripts/tiny_overfit_ctc.py `
  --manifest outputs/ctc-audit/tiny_manifest.csv `
  --processor-path C:\path\to\validated_processor `
  --output-dir outputs/tiny-overfit `
  --trainable-encoder-layers 4 `
  --head-lr 1e-3 --encoder-lr 1e-5 `
  --batch-size 1 --gradient-accumulation-steps 8 `
  --max-steps 3000 --eval-steps 25 `
  --seed 42 --fp16 --gradient-checkpointing `
  --manually-verified
```

The script refuses to start without CUDA unless `--allow-cpu` is explicit. Use
`--dry-run` first, then `--smoke-test`, and only then launch the full 3,000-step
run. It writes the locked manifest, processor/model/environment audits,
step-0 and periodic predictions, raw token traces, checkpoints, JSONL history,
and an atomic status file under the requested output directory.

## Full training is experimental

`scripts/train_research.py` and `configs/research.yaml` are retained for model
development and reproducibility, not as the recommended next run. The current
schedule freezes the encoder throughout phase 1 while training randomly
initialized CTC/MoE/dialect components. The MoE also broadcasts an
utterance-level expert vector across every time frame. Both are under review.

If the plain tiny test passes, the experiment order is:

1. plain MMS-CTC baseline with the top 2–4 transformer layers trainable from
   step 0;
2. full-data plain baseline selected by validation WER;
3. add a low-weight dialect head and confirm ASR does not regress; and
4. add a corrected, gated or frame-wise MoE last.

Never resume the collapsed checkpoint into a changed architecture or schedule.
Start a new output directory and record the new configuration.

## Development

Run the maintained test suite:

```bash
python -m pytest -q
```

Large datasets, checkpoints, notebook outputs, logs, and credentials must not be
committed. MMS-300M is used here for noncommercial research; verify all upstream
licenses and obtain Bengali linguistic review before publishing dialect claims.
