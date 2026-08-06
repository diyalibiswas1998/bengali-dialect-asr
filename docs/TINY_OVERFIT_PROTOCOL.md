# Plain MMS-CTC tiny-overfit protocol

This is a debugging gate for the Bengali CTC pipeline. It is not a
generalization benchmark and it must not load the failed MoE checkpoint.

## Contract

`scripts/tiny_overfit_ctc.py` loads a fresh `facebook/mms-300m` acoustic
backbone and initializes a new 73-way CTC head. It freezes the convolutional
feature encoder and trains the configurable top transformer layers (four by
default) plus the head. The optimizer has separate head and encoder learning
rates, constant learning rates after a short warm-up, and no distributed mode.

The input manifest must contain exactly 32 rows, each marked
`manually_verified=YES`. The same rows are used for optimization and
evaluation. The supplied processor is audited before any model download:

```text
vocabulary size = 73
blank/pad ID   = 0
unknown ID     = 1
delimiter ID   = 2 (|)
sampling rate  = 16000 Hz
normalization  = enabled
```

The script refuses a missing or mismatched processor and has no checkpoint
argument. The failed MoE weights cannot be loaded accidentally.

## Recommended sequence

1. Generate a candidate 32-row manifest with the existing diagnostics helper.
2. Listen to every pair and set `manually_verified` to `YES` only after the
   audio and transcript agree.
3. Run `--dry-run` to validate audio, targets, model shapes, and one CTC loss.
4. Run `--smoke-test` for three optimizer steps.
5. Run the complete test for up to 3,000 optimizer steps.

Example (Linux/macOS):

```bash
python scripts/tiny_overfit_ctc.py \
  --manifest outputs/ctc-audit/tiny_manifest.csv \
  --processor-path /path/to/validated_processor \
  --output-dir outputs/tiny-overfit \
  --trainable-encoder-layers 4 \
  --head-lr 1e-3 --encoder-lr 1e-5 \
  --batch-size 1 --gradient-accumulation-steps 8 \
  --max-steps 3000 --eval-steps 25 \
  --seed 42 --fp16 --gradient-checkpointing \
  --manually-verified
```

On Windows, keep `--num-workers 0` (the default) and use PowerShell backticks
for line continuation. Set `CUDA_VISIBLE_DEVICES=0` before launching if more
than one GPU is visible. The script uses only `torch.device("cuda")` and never
initializes DDP, DataParallel, Accelerate, or DeepSpeed.

## Gate decision

The status is `passed` only if the same evaluation checkpoint satisfies all
three conditions:

```text
CER <= 0.05
WER <= 0.05
empty_prediction_rate < 0.10
```

The run also records blank/delimiter/unknown frame fractions, probabilities,
entropy, prediction lengths, CTC lengths and repeated-label requirements,
gradient norms, learning rates, parameter-update checks, GPU memory, step
times, and raw token traces. A low loss alone never passes the gate.

## Output contract

The requested output directory contains `run_config.json`, `environment.json`,
`processor_audit.json`, `model_audit.json`, the locked manifest and metadata,
`tiny_target_audit.csv`, `tiny_overfit_status.json`, JSONL history, atomic
`tiny_overfit_best.pt`/`tiny_overfit_last.pt` checkpoints, `ctc_lengths_audit.jsonl`, prediction snapshots,
raw token traces, `ctc_collapse_summary.json`, and both `logs/tiny_overfit.log`
and `logs/checkpoint_audit.log`.

If CUDA is unavailable, the program stops unless `--allow-cpu` is explicit.
CPU mode is suitable for contract checks only; it is not evidence that
MMS-300M can complete the 3,000-step experiment on the laptop.
