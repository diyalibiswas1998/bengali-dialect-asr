# Bengali MMS CTC collapse audit

This audit targets the failed Kaggle run built from commit
`21b7519673647285380de23a0f4556812430be78`.  The run was not restarted and no
full 6,000-step experiment was launched while preparing this fix.

## Files and execution path

| Concern | Exact-commit location | Failed diagnostic trainer location |
|---|---|---|
| Custom model | `src/asr_dialect_benchmark/modeling/asr_model.py`, `BengaliDialectASR` | `scripts/trainer_mms_ctc_fixed.py` imports the same class |
| Bengali CTC head | `BengaliDialectASR.ctc_head = nn.Linear(hidden_size, num_tokens)` | `num_tokens` is built from the 73-token MMS processor |
| CTC loss | `src/asr_dialect_benchmark/losses/ctc_losses.py::multitask_loss` | `scripts/trainer_mms_ctc_fixed.py::multitask_loss` |
| Labels/collator | `src/asr_dialect_benchmark/data/processed_vaani.py::processed_collate` | `scripts/trainer_mms_ctc_fixed.py::CTCDataCollator` |
| Training phases | `scripts/train_research.py`, the `for phase in range(..., 4)` loop | same three-phase loop in the diagnostic trainer |
| Existing diagnostics | progress logging in the diagnostic trainer | new `scripts/ctc_collapse_diagnostics.py` |

The Bengali path is:

```text
MMS encoder -> optional MoE/router -> Bengali ctc_head(73)
             -> CTC logits -> log_softmax -> CTCLoss(blank=0)
```

The MMS/Dia internal `encoder_config.vocab_size=256` and
`decoder_config.vocab_size=1028` are not the Bengali CTC output space.  The
decoder pad ID `1025` is never a valid CTC blank and is not used by the patch.

## Evidence from the failed run

The supplied log reports all of the following before training starts:

```text
ctc_blank_id=0, padding_id=0, unknown_id=1, delimiter_id=2
vocabulary_size=73
feature sampling_rate=16000, do_normalize=True
validation/test unknown rate=0.0
two Tesla T4 GPUs
```

At steps 200–2,000, the raw frame diagnostics repeatedly report
`delimiter_fraction=1.0000`, `blank_fraction=0.0000`, and
`empty_prediction_rate=1.0000`.  Therefore this log demonstrates genuine
delimiter-token collapse in the raw argmax logits; it does **not** demonstrate
CTC blank collapse.  The old validation crash happened after checkpoint step
2,000 and is independent:

```text
float(row.diag().item() / denominator)
RuntimeError: a Tensor with 16 elements cannot be converted to Scalar
```

`row` was already a one-dimensional confusion-matrix row.  The prior notebook
fix replaces that expression with `confusion[class_index, class_index].item()`.

## CTC-loss audit

The exact-commit generic loss already calls `F.ctc_loss(..., blank=0,
zero_infinity=True)`.  The failed diagnostic trainer passed the processor pad
ID into the loss; the run log proves that value was 0.  No search of the model
or trainer found `masked_fill`, `-inf`, or `-1e9` applied to output logit ID 0.
The only target padding operation was in the collator.  The failed trainer
then sliced each row to `target_lengths` before flattening, so padding was not
actually sent to CTCLoss; the new trainer still uses `-100` as the visible
padding sentinel and explicitly removes it by length.

The patch now enforces, immediately before the loss:

```python
assert ctc_blank_id == 0
assert logits.ndim == 3
assert logits.shape[-1] == 73
```

It prints the blank ID, logits shape, and CTC vocabulary once, rejects blank ID
0 in valid targets, and rejects any sample whose encoder length is below
`target_length + adjacent_repeated_target_count`.  This prevents
`zero_infinity=True` from silently hiding an invalid alignment.

## Confirmed, ruled out, and unresolved causes

### Confirmed

1. The observed collapse is delimiter dominance, not blank dominance.
2. The failed run used the intended 73-way Bengali CTC vocabulary and blank ID
   0 according to its own processor metadata and token printout.
3. MMS waveform normalization and 16-kHz processing were enabled.
4. The failure was not a T4/DDP or Hugging Face authentication failure;
   training reached step 2,000 on both T4 ranks and wrote checkpoints.
5. The validation crash was a separate one-dimensional confusion-row bug.

### Ruled out by code/log evidence

1. Decoder vocabulary 1028 and decoder pad ID 1025 being used as CTC IDs.
2. A missing Bengali character in validation/test transcripts (unknown rate was
   zero in the run metadata).
3. Explicit suppression of blank output logit ID 0.

### Still requiring measurement

1. The initial/early/final CTC-head bias values are not present in the supplied
   log.  The checkpoint auditor writes them for every attached checkpoint.
2. The exact valid-target ID distribution and per-sample CTC length margin were
   not logged by the failed run.  The auditor reports both.
3. The 2,000-step frozen-encoder phase plus randomly initialized MoE/CTC head
   is a plausible contributor to learned delimiter collapse, but it is not
   proven as the root cause until the plain 32-sample test is run.

## Necessary fixes

`scripts/trainer_mms_ctc_fixed.py` now:

- hard-requires the 73-token processor, `<pad>/<blank>=0`, `<unk>=1`, and
  delimiter `|=2`;
- pads batch targets with `-100` and flattens only valid target lengths;
- asserts the CTC head/logit shape and blank ID directly before CTCLoss;
- uses the literal `blank=0` in CTCLoss;
- checks target IDs and CTC minimum lengths before `zero_infinity` can hide an
  error; and
- prints the contract exactly once per process.

The standalone tools are:

- `scripts/ctc_collapse_diagnostics.py`: checkpoint/validation audit with raw
  token counts, label audit, length audit, bias values, predictions, CER/WER,
  and CSV/JSON output;
- `scripts/tiny_overfit_ctc.py`: one-GPU, plain MMS-CTC 20–50-example overfit
  test with no MoE, dialect loss, augmentation, or distributed training.

The tiny test requires a manually verified manifest and uses the same 73-token
processor.  A passing result means non-empty predictions and near-zero CER/WER
on the same examples.  A failing result means the full MoE experiment must not
be resumed.

## Kaggle/CLI commands

Clone the immutable diagnostic revision:

```bash
git clone https://github.com/diyalibiswas1998/bengali-dialect-asr.git
cd bengali-dialect-asr
git checkout d58512c
```

Checkpoint audit (read-only; run in Kaggle with Internet enabled and the dataset/checkpoint attached):

```bash
python scripts/ctc_collapse_diagnostics.py \
  --data-root /kaggle/input/four-dialect-data-undersampled \
  --repo-root /kaggle/working/bengali-dialect-asr \
  --checkpoint /kaggle/input/<checkpoint-dataset>/<checkpoint-dir> \
  --output-dir /kaggle/working/ctc-collapse-diagnostics \
  --sample-count 100 --batch-size 4
```

Create the tiny manifest:

```bash
python scripts/ctc_collapse_diagnostics.py \
  --data-root /kaggle/input/four-dialect-data-undersampled \
  --repo-root /kaggle/working/bengali-dialect-asr \
  --output-dir /kaggle/working/ctc-collapse-diagnostics \
  --make-manifest /kaggle/working/ctc-collapse-diagnostics/tiny_manifest.csv \
  --manifest-count 32
```

After listening to all rows and changing `manually_verified` to `YES`, run the
one-GPU plain-CTC test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tiny_overfit_ctc.py \
  --manifest /kaggle/working/ctc-collapse-diagnostics/tiny_manifest.csv \
  --checkpoint /kaggle/input/<checkpoint-dataset>/<checkpoint-dir> \
  --output-dir /kaggle/working/ctc-collapse-diagnostics/tiny-overfit \
  --batch-size 4 --max-steps 3000 --eval-every 50 --manually-verified
```

The diagnostic-only notebook is `kaggle_upload/ctc_collapse_diagnostics/
bengali_ctc_collapse_diagnostics.ipynb`. Kaggle could not publish its GPU
version while the account was at the 30-hour weekly GPU quota; retry the normal
push after quota reset:

```bash
python -m kaggle kernels push \
  -p kaggle_upload/ctc_collapse_diagnostics \
  --accelerator NvidiaTeslaT4
```