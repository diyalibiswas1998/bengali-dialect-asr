# CTC delimiter-collapse incident

## Decision status

The existing Bengali MMS-MoE checkpoint is not a usable ASR model. Full MoE
training is paused until a plain MMS-CTC model can overfit 32 manually verified
audio/transcript pairs.

This document separates measured facts from hypotheses. A proposed change such
as unfreezing encoder layers in phase 1 must not be described as a fix until the
plain baseline gate passes.

## What was measured

The phase-3 checkpoint was loaded from `model.safetensors` with no missing or
unexpected project weights. Its saved processor and 73-way Bengali CTC head
agreed on the following IDs:

```text
blank/pad = 0
unknown   = 1
delimiter = 2 (|)
vocabulary size = 73
sampling rate = 16,000 Hz
waveform normalization = enabled
```

The checkpoint audit used 100 validation utterances and reported:

| Check | Result |
|---|---:|
| Valid target tokens | 6,655 |
| Blank IDs in valid targets | 0 |
| Unknown IDs in valid targets | 0 |
| Delimiter IDs in valid targets | 1,105 (16.60%) |
| Target decode mismatches | 0 |
| Invalid CTC length relationships | 0 |
| Blank argmax fraction | 0.0000 |
| Delimiter argmax fraction | 0.95884 |
| Mean blank probability | 0.00294 |
| Empty decoded prediction rate | 0.04 |
| CER | 0.91874 |
| WER | 1.00000 |

The most common frame token was ID 2, with 26,114 of 27,235 audited frames.
Therefore the observed failure is delimiter dominance. Calling it “blank
collapse” is inaccurate.

All four dialect groups had WER 1.0 in this sample. The failure is global rather
than isolated to one minority dialect.

## What has been ruled out

The evidence does not support these explanations:

- CTC used the MMS decoder pad ID instead of Bengali blank ID 0.
- Bengali validation targets contained unknown tokens.
- Target padding (`-100`) was passed into the valid CTC target sequence.
- Encoder output lengths were too short for the target and repeated labels.
- MMS waveform normalization or 16-kHz resampling was disabled.
- The project CTC head was absent from the loaded checkpoint.
- A Hugging Face authentication or T4/DDP failure caused the trained output.

The old confusion-matrix exception and later self-contained-notebook packaging
errors were separate evaluation/runtime defects. They did not cause the model's
delimiter-heavy logits.

## Highest-risk model behavior

The current MoE is routed from an utterance-level pooled vector:

```text
frame states
    -> masked mean over time
    -> top-k residual dialect experts + residual shared expert
    -> one fused utterance vector
    -> broadcast the same vector to every time frame
    -> CTC head
```

In code, the final operation is equivalent to:

```python
hidden_states + fusion.unsqueeze(1)
```

Each expert block already returns a residual-transformed pooled vector. The MoE
then adds the expert mixture and shared expert and applies another outer
residual. This can introduce a large, time-invariant shift into every CTC frame,
reducing the relative acoustic variation that CTC needs. A global token prior
such as the word delimiter can then dominate every frame.

This is a plausible mechanism consistent with the measured logits, not yet a
proven single root cause.

## Training-schedule risks

Three schedule choices can reinforce the architecture risk:

1. Phase 1 freezes the entire MMS encoder for 2,000 optimizer steps while the
   random CTC head, router, four experts, shared expert, and dialect classifier
   train together.
2. The encoder optimizer group follows the global linear schedule while frozen,
   so delayed unfreezing begins after part of its useful schedule has elapsed.
3. Dialect loss weight `0.2` is active before speech-to-text alignment has been
   demonstrated.

Unfreezing the top 2–4 transformer layers from step 0 is reasonable for a plain
CTC baseline. It is not sufficient as a standalone modification to the current
MoE run: the encoder could simply adapt to the same collapsed multitask
objective.

## Required experiment ladder

### Gate 1: 32-sample plain MMS-CTC overfit

- Fresh `facebook/mms-300m` model.
- Same validated 73-token processor.
- Same 32 manually checked examples for training and evaluation.
- No MoE, router, dialect classifier loss, load balancing, or augmentation.
- Convolutional feature encoder frozen.
- Train the CTC head and transformer layers with separate learning rates.
- Log raw blank and delimiter fractions, prediction length, empty rate, CER,
  WER, head gradient norm, encoder gradient norm, and learning rates.

Pass criteria used by `scripts/tiny_overfit_ctc.py`:

```text
best CER <= 0.05
best WER <= 0.05
best empty-prediction rate < 0.10
```

If this fails, stop. The fundamental plain-ASR pipeline remains broken.

### Gate 2: full-data plain MMS-CTC baseline

Train a plain baseline for complete dataset passes, not another short MoE run.
Keep the top 2–4 transformer layers trainable from step 0, maintain separate
head/encoder learning rates, and select the best checkpoint by validation WER.

### Gate 3: dialect head

Add dialect classification with a small initial weight such as `0.02`. Continue
only if validation ASR does not materially regress.

### Gate 4: corrected MoE

Route using an utterance representation if desired, but transform frame-level
states or introduce a small residual gate initialized near zero. Avoid injecting
an unrestricted pooled residual identically into every CTC frame. Add the shared
expert and top-k routing through explicit ablations.

## Files to inspect after the tiny test

```text
tiny-overfit/tiny_overfit_status.json
tiny-overfit/tiny_overfit_history.jsonl
tiny-overfit/tiny_overfit_best.pt
ctc_collapse_summary.json
ctc_predictions_<checkpoint>.csv
logs/checkpoint_audit.log
```

The history should show whether delimiter fraction falls while CER/WER improve.
A low loss alone is not a pass condition.
