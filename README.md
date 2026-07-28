# Bengali Dialect ASR with Sparse Mixture-of-Experts

This repository implements a research-grade Bengali dialect-aware speech-to-text framework that combines a pretrained speech foundation encoder with a sparse mixture-of-experts (MoE) module and a dialect router. The design supports four Bengali dialects and uses a three-stage training recipe:

1. Router-only training with frozen encoder.
2. Joint training of experts, shared expert, and CTC decoder with dialect and load-balancing losses.
3. End-to-end fine-tuning with unfreezed top encoder layers.

## Highlights

- Pretrained encoder backbone via Hugging Face Transformers (XLS-R 300M or WavLM Large)
- Dialect routing network with top-2 expert selection
- Shared expert + dialect experts with residual feed-forward blocks
- CTC-based transcription head
- Hydra configuration, mixed precision, gradient accumulation, early stopping, TensorBoard logging, and checkpointing
- Evaluation metrics: WER, CER, MER, WIL, WIP, per-dialect WER, confusion matrix, router accuracy, expert utilization, and load balancing statistics

## Project layout

- configs/: Hydra configuration files
- data/: example manifest and data assets
- scripts/: training, validation, and inference entry points
- src/asr_dialect_benchmark/: modular implementation package

## Quick start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare a manifest file in JSONL format with fields:

   ```json
   {"audio_path": "path/to/audio.wav", "transcript": "আপনার টেক্সট", "dialect_label": "barishal"}
   ```

3. Train:

   ```bash
   python scripts/train.py data.manifest_path=/path/to/train.jsonl
   ```

4. Validate:

   ```bash
   python scripts/validate.py ckpt_path=/path/to/checkpoint.pt
   ```

5. Infer:

   ```bash
   python scripts/infer.py ckpt_path=/path/to/checkpoint.pt manifest_path=/path/to/test.jsonl output_path=/tmp/preds.jsonl
   ```

## Configuration knobs

Ablation switches are available in the configuration files to disable:

- Router
- Shared Expert
- Load Balancing
- Top-k Routing
- Dialect Loss
