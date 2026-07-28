"""
train_vaani.py
==============
Self-contained training entry-point that uses ``VaaniDataset`` (direct
HuggingFace streaming) for the 11 West Bengal district configs.

This script does NOT rely on JSONL manifests or disk audio files.
It builds all datasets, loaders, model, and optimizer from scratch so there
are no dependency conflicts with the original JSONL-based Trainer.

Usage (from research/code/ directory)
--------------------------------------
  python scripts/train_vaani.py

  # Override Hydra config values:
  python scripts/train_vaani.py training.batch_size=4 training.max_epochs=5
  python scripts/train_vaani.py vaani.streaming=true training.device=cpu
"""

import sys
from functools import partial
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_dialect_benchmark.common.utils import ensure_dir, seed_everything
from asr_dialect_benchmark.data.vaani_dataset import VaaniDataset, vaani_collate_batch, VAANI_CONFIGS
from asr_dialect_benchmark.losses.ctc_losses import LoadBalancingLoss
from asr_dialect_benchmark.modeling.asr_model import BengaliDialectASR
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _step_batch(model, batch, device, cfg, criterion, load_balancing_criterion, scaler):
    """Run one forward pass and return (total_loss, class_loss, ctc_loss)."""
    audio = batch["audio"].to(device)
    dialect_labels = batch["dialect_label"].to(device)
    target = batch["target"].to(device)
    audio_length = batch["audio_length"].to(device)
    target_length = batch["target_length"].to(device)

    enabled = scaler.is_enabled()
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    with autocast(device_type=amp_device, enabled=enabled):
        outputs = model(audio, attention_mask=None, labels=target, dialect_labels=dialect_labels)
        logits = outputs["logits"]
        dialect_logits = outputs["dialect_logits"]

        ctc_logits = logits.transpose(0, 1)
        # Subsampled frame length output by Wav2Vec2 encoder
        input_lengths = model.encoder._get_feat_extract_output_lengths(audio_length)
        input_lengths = torch.clamp(input_lengths, max=ctc_logits.size(0))

        ctc_loss = model.loss_fn(
            ctc_logits.log_softmax(-1), target, input_lengths, target_length
        )
        class_loss = criterion(dialect_logits, dialect_labels)
        total_loss = class_loss + ctc_loss

        if cfg.loss.use_load_balancing and cfg.model.use_router:
            gate_probs = torch.softmax(dialect_logits, dim=-1)
            top_k = getattr(cfg.model, "top_k", 2)
            topk_vals, topk_indices = model.moe.topk_gating(gate_probs, top_k)
            total_loss = total_loss + load_balancing_criterion(gate_probs, topk_indices)

    return total_loss, class_loss, ctc_loss


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path=str(ROOT / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(42)
    ensure_dir(cfg.training.output_dir)
    ensure_dir(cfg.training.log_dir)

    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    print(f"\n[train_vaani] Device: {device}")

    # ── Vaani config knobs ───────────────────────────────────────────────────
    vaani_cfg = cfg.get("vaani", {})
    hf_token  = vaani_cfg.get("hf_token", None)
    max_dur   = vaani_cfg.get("max_duration", 30.0)
    max_samp  = vaani_cfg.get("max_samples_per_config", None)  # None = all data
    configs   = list(vaani_cfg.get("configs", []))  # empty → all 11 districts

    # ── Tokenizer ────────────────────────────────────────────────────────────
    # Fit the tokenizer on training transcripts first so vocab is ready.
    tokenizer = SimpleTokenizer()

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\n[train_vaani] Building training dataset ...")
    train_ds = VaaniDataset(
        configs=configs or None,
        split="train",
        hf_token=hf_token,
        tokenizer=tokenizer,
        max_duration=max_dur,
        max_samples_per_config=max_samp,
    )

    # Update cfg num_tokens to match the fitted vocabulary size so the
    # ctc_head dimension is correct.
    vocab_size = len(tokenizer.vocab)
    print(f"[train_vaani] Tokenizer vocab size: {vocab_size}")
    # Override config value (omegaconf struct mode requires unlock)
    from omegaconf import OmegaConf
    OmegaConf.update(cfg, "model.num_tokens", vocab_size, merge=True)

    print("\n[train_vaani] Building validation dataset ...")
    val_ds = VaaniDataset(
        configs=configs or None,
        split="validation",
        hf_token=hf_token,
        tokenizer=tokenizer,
        max_duration=max_dur,
        max_samples_per_config=max_samp,
    )

    collate_fn = partial(vaani_collate_batch, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,   # IterableDataset does not support shuffle
        num_workers=0,   # must be 0 for IterableDataset on Windows
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"[train_vaani] Streaming {len(configs or VAANI_CONFIGS)} district configs | batch_size={cfg.training.batch_size}")

    # ── Model / optimiser ────────────────────────────────────────────────────
    print("\n[train_vaani] Building model ...")
    model = BengaliDialectASR(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    criterion = torch.nn.CrossEntropyLoss()
    lb_criterion = LoadBalancingLoss(importance_weight=cfg.loss.load_balancing_weight)
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = GradScaler(device=amp_device, enabled=cfg.training.use_amp and torch.cuda.is_available())
    writer = SummaryWriter(log_dir=cfg.training.log_dir)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train_vaani] Model parameters: {num_params:.1f}M")

    best_val_loss = float("inf")
    global_step = 0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(cfg.training.max_epochs):
        model.train()
        # tqdm without total — IterableDataset has no len()
        pbar = tqdm(desc=f"Epoch {epoch+1}/{cfg.training.max_epochs}", unit="batch")
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, class_loss, ctc_loss = _step_batch(
                model, batch, device, cfg, criterion, lb_criterion, scaler
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/class_loss", class_loss.item(), global_step)
            writer.add_scalar("train/ctc_loss", ctc_loss.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)
        pbar.close()

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                v_loss, _, _ = _step_batch(
                    model, batch, device, cfg, criterion, lb_criterion, scaler
                )
                total_val_loss += v_loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / max(1, val_batches)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        print(f"[Epoch {epoch+1}] val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = Path(cfg.training.output_dir) / f"vaani_ckpt_epoch{epoch+1}.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch + 1}, ckpt_path)
            writer.add_scalar("val/best_loss", best_val_loss, epoch)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    writer.close()
    print(f"\n[train_vaani] Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
