"""Generate the Kaggle .ipynb notebook file."""
import json

notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python", "isGpuEnabled": True}
    },
    "cells": []
}

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None}

cells = notebook["cells"]

# --- Markdown header ---
cells.append(md([
    "# Bengali Dialect ASR — Sparse MoE | Kaggle GPU Training\n",
    "\n",
    "This notebook clones the codebase from GitHub and trains the **BengaliDialectASR**\n",
    "(Sparse Mixture-of-Experts) model on the **ARTPARK-IISc/Vaani** dataset\n",
    "(11 West Bengal districts) using Kaggle Tesla T4 GPU.\n",
    "\n",
    "### Setup Checklist\n",
    "- [x] Accelerator: **GPU T4 x2**\n",
    "- [x] Internet: **Enabled**\n",
    "\n",
    "### Architecture\n",
    "- **Encoder**: Wav2Vec2 (6-layer Transformer, 768-dim)\n",
    "- **MoE**: 11 Dialect Experts + 1 Shared Expert, Top-K=2 routing\n",
    "- **Heads**: CTC ASR + Dialect Classifier\n",
    "- **Parameters**: ~61.7M\n",
]))

# --- Cell 1: Install deps ---
cells.append(code([
    "# Cell 1: Install dependencies\n",
    "import subprocess, sys, os\n",
    "\n",
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n",
    "    'datasets', 'huggingface_hub', 'transformers', 'soundfile',\n",
    "    'tensorboard', 'tqdm', 'hydra-core', 'omegaconf'])\n",
    "\n",
    "print('All dependencies installed!')\n",
]))

# --- Cell 2: Clone repo ---
cells.append(code([
    "# Cell 2: Set tokens and clone repository\n",
    "import os, sys\n",
    "\n",
    '# HuggingFace token for Vaani dataset access (set HF_TOKEN env var)\n',
    'if "HF_TOKEN" not in os.environ:\n',
    '    os.environ["HF_TOKEN"] = input("Please enter your HuggingFace Token: ") if sys.stdin.isatty() else ""\n',
    "\n",
    "# Clone the codebase from GitHub\n",
    'REPO_URL = "https://github.com/diyalibiswas1998/bengali-dialect-asr.git"\n',
    'CLONE_DIR = "/kaggle/working/bengali-dialect-asr"\n',
    "\n",
    "if not os.path.exists(CLONE_DIR):\n",
    "    !git clone {REPO_URL} {CLONE_DIR}\n",
    "else:\n",
    "    !git -C {CLONE_DIR} pull\n",
    "\n",
    'sys.path.insert(0, os.path.join(CLONE_DIR, "src"))\n',
    'print("Repository ready!")\n',
]))

# --- Cell 3: Check GPU ---
cells.append(code([
    "# Cell 3: Check GPU\n",
    "import torch\n",
    "\n",
    'print(f"CUDA Available: {torch.cuda.is_available()}")\n',
    "if torch.cuda.is_available():\n",
    '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n',
    '    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9\n',
    '    print(f"VRAM: {mem_gb:.1f} GB")\n',
    "else:\n",
    '    print("WARNING: No GPU! Enable in Settings > Accelerator")\n',
]))

# --- Cell 4: Load config ---
cells.append(code([
    "# Cell 4: Load and override config for Kaggle T4\n",
    "from omegaconf import OmegaConf\n",
    "\n",
    'config_path = os.path.join(CLONE_DIR, "configs", "config.yaml")\n',
    "cfg = OmegaConf.load(config_path)\n",
    "\n",
    "# --- Kaggle GPU overrides ---\n",
    'OmegaConf.update(cfg, "training.device", "cuda", merge=True)\n',
    'OmegaConf.update(cfg, "training.use_amp", True, merge=True)\n',
    'OmegaConf.update(cfg, "training.batch_size", 32, merge=True)\n',
    'OmegaConf.update(cfg, "training.max_epochs", 3, merge=True)\n',
    'OmegaConf.update(cfg, "training.lr", 2e-4, merge=True)\n',
    'OmegaConf.update(cfg, "training.log_dir", "/kaggle/working/logs", merge=True)\n',
    'OmegaConf.update(cfg, "training.output_dir", "/kaggle/working/outputs", merge=True)\n',
    'OmegaConf.update(cfg, "training.num_workers", 2, merge=True)\n',
    'OmegaConf.update(cfg, "vaani.max_samples_per_config", None, merge=True)\n',
    'OmegaConf.update(cfg, "vaani.hf_token", os.environ["HF_TOKEN"], merge=True)\n',
    "\n",
    'print("Config overrides applied:")\n',
    "print(OmegaConf.to_yaml(cfg))\n",
]))

# --- Cell 5: Imports ---
cells.append(code([
    "# Cell 5: Import modules from cloned codebase\n",
    "import sys, os\n",
    'CLONE_DIR = "/kaggle/working/bengali-dialect-asr"\n',
    'sys.path.insert(0, os.path.join(CLONE_DIR, "src"))\n',
    "\n",
    "from functools import partial\n",
    "from pathlib import Path\n",
    "\n",
    "import numpy as np\n",
    "import torch.nn as nn\n",
    "from torch.amp import GradScaler, autocast\n",
    "from torch.utils.data import DataLoader\n",
    "from torch.utils.tensorboard import SummaryWriter\n",
    "from tqdm import tqdm\n",
    "\n",
    "from asr_dialect_benchmark.common.utils import ensure_dir, seed_everything\n",
    "from asr_dialect_benchmark.data.vaani_dataset import VaaniDataset, vaani_collate_batch\n",
    "from asr_dialect_benchmark.losses.ctc_losses import LoadBalancingLoss\n",
    "from asr_dialect_benchmark.modeling.asr_model import BengaliDialectASR\n",
    "from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer\n",
    "\n",
    'print("All modules imported!")\n',
]))

# --- Cell 6: Training function ---
cells.append(code([
    "# Cell 6: Define training step function\n",
    "\n",
    "def _step_batch(model, batch, device, cfg, criterion, lb_criterion, scaler):\n",
    '    audio = batch["audio"].to(device)\n',
    '    dialect_labels = batch["dialect_label"].to(device)\n',
    '    target = batch["target"].to(device)\n',
    '    audio_length = batch["audio_length"].to(device)\n',
    '    target_length = batch["target_length"].to(device)\n',
    "\n",
    "    enabled = scaler.is_enabled()\n",
    '    with autocast(device_type="cuda", enabled=enabled):\n',
    "        outputs = model(audio, attention_mask=None, labels=target, dialect_labels=dialect_labels)\n",
    '        logits = outputs["logits"]\n',
    '        dialect_logits = outputs["dialect_logits"]\n',
    "\n",
    "        ctc_logits = logits.transpose(0, 1)\n",
    "        input_lengths = model.encoder._get_feat_extract_output_lengths(audio_length)\n",
    "        input_lengths = torch.clamp(input_lengths, max=ctc_logits.size(0))\n",
    "\n",
    "        ctc_loss = model.loss_fn(\n",
    "            ctc_logits.log_softmax(-1), target, input_lengths, target_length\n",
    "        )\n",
    "        class_loss = criterion(dialect_logits, dialect_labels)\n",
    "        total_loss = class_loss + ctc_loss\n",
    "\n",
    "        if cfg.loss.use_load_balancing and cfg.model.use_router:\n",
    "            gate_probs = torch.softmax(dialect_logits, dim=-1)\n",
    '            top_k = getattr(cfg.model, "top_k", 2)\n',
    "            topk_vals, topk_indices = model.moe.topk_gating(gate_probs, top_k)\n",
    "            total_loss = total_loss + lb_criterion(gate_probs, topk_indices)\n",
    "\n",
    "    return total_loss, class_loss, ctc_loss\n",
    "\n",
    'print("Training step function defined!")\n',
]))

# --- Cell 7: Main training ---
cells.append(code([
    "# Cell 7: Launch Training\n",
    "\n",
    "seed_everything(42)\n",
    "ensure_dir(cfg.training.output_dir)\n",
    "ensure_dir(cfg.training.log_dir)\n",
    "\n",
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n',
    "\n",
    "# --- Config ---\n",
    'vaani_cfg = cfg.get("vaani", {})\n',
    'hf_token = vaani_cfg.get("hf_token", None)\n',
    'max_dur = vaani_cfg.get("max_duration", 30.0)\n',
    'max_samp = vaani_cfg.get("max_samples_per_config", None)\n',
    'configs = list(vaani_cfg.get("configs", []))\n',
    "\n",
    "tokenizer = SimpleTokenizer()\n",
    "\n",
    "# --- Datasets ---\n",
    'print("\\n[train] Building training dataset ...")\n',
    "train_ds = VaaniDataset(\n",
    "    configs=configs or None, split='train', hf_token=hf_token,\n",
    "    tokenizer=tokenizer, max_duration=max_dur, max_samples_per_config=max_samp,\n",
    ")\n",
    "\n",
    "vocab_size = len(tokenizer.vocab)\n",
    'OmegaConf.update(cfg, "model.num_tokens", vocab_size, merge=True)\n',
    "\n",
    'print("[train] Building validation dataset ...")\n',
    "val_ds = VaaniDataset(\n",
    "    configs=configs or None, split='validation', hf_token=hf_token,\n",
    "    tokenizer=tokenizer, max_duration=max_dur, max_samples_per_config=max_samp,\n",
    ")\n",
    "\n",
    "collate_fn = partial(vaani_collate_batch, tokenizer=tokenizer)\n",
    "train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,\n",
    "    shuffle=False, num_workers=cfg.training.num_workers,\n",
    "    collate_fn=collate_fn, pin_memory=True)\n",
    "val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size,\n",
    "    shuffle=False, num_workers=0, collate_fn=collate_fn)\n",
    "\n",
    "# --- Model ---\n",
    'print("[train] Building model ...")\n',
    "model = BengaliDialectASR(cfg).to(device)\n",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "lb_criterion = LoadBalancingLoss(importance_weight=cfg.loss.load_balancing_weight)\n",
    'scaler = GradScaler(device="cuda", enabled=cfg.training.use_amp and torch.cuda.is_available())\n',
    "writer = SummaryWriter(log_dir=cfg.training.log_dir)\n",
    "\n",
    "num_params = sum(p.numel() for p in model.parameters()) / 1e6\n",
    'print(f"[train] Model: {num_params:.1f}M params | batch_size={cfg.training.batch_size} | AMP={cfg.training.use_amp}")\n',
    "\n",
    'best_val_loss = float("inf")\n',
    "global_step = 0\n",
    "\n",
    "# --- Training loop ---\n",
    "for epoch in range(cfg.training.max_epochs):\n",
    "    model.train()\n",
    '    pbar = tqdm(desc=f"Epoch {epoch+1}/{cfg.training.max_epochs}", unit="batch")\n',
    "    for batch in train_loader:\n",
    "        optimizer.zero_grad(set_to_none=True)\n",
    "        loss, class_loss, ctc_loss = _step_batch(\n",
    "            model, batch, device, cfg, criterion, lb_criterion, scaler)\n",
    "        scaler.scale(loss).backward()\n",
    "        scaler.step(optimizer)\n",
    "        scaler.update()\n",
    '        writer.add_scalar("train/loss", loss.item(), global_step)\n',
    "        global_step += 1\n",
    '        pbar.set_postfix(loss=f"{loss.item():.4f}")\n',
    "        pbar.update(1)\n",
    "    pbar.close()\n",
    "\n",
    "    # Validation\n",
    "    model.eval()\n",
    "    total_val_loss, val_batches = 0.0, 0\n",
    "    with torch.no_grad():\n",
    "        for batch in val_loader:\n",
    "            v_loss, _, _ = _step_batch(model, batch, device, cfg, criterion, lb_criterion, scaler)\n",
    "            total_val_loss += v_loss.item()\n",
    "            val_batches += 1\n",
    "    avg_val_loss = total_val_loss / max(1, val_batches)\n",
    '    print(f"\\n[Epoch {epoch+1}] val_loss={avg_val_loss:.4f}")\n',
    "\n",
    "    if avg_val_loss < best_val_loss:\n",
    "        best_val_loss = avg_val_loss\n",
    '        ckpt_path = Path(cfg.training.output_dir) / f"vaani_ckpt_epoch{epoch+1}.pt"\n',
    "        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),\n",
    "                    'epoch': epoch+1, 'global_step': global_step,\n",
    "                    'best_val_loss': best_val_loss}, ckpt_path)\n",
    '        print(f"  -> Saved: {ckpt_path}")\n',
    "\n",
    "writer.close()\n",
    'print(f"\\nDone! Best val loss: {best_val_loss:.4f}")\n',
]))

# --- Cell 8: List checkpoints ---
cells.append(code([
    "# Cell 8: List saved checkpoints\n",
    "import os\n",
    "ckpt_dir = cfg.training.output_dir\n",
    "for f in sorted(os.listdir(ckpt_dir)):\n",
    "    fpath = os.path.join(ckpt_dir, f)\n",
    "    size_mb = os.path.getsize(fpath) / 1e6\n",
    '    print(f"  {f} ({size_mb:.1f} MB)")\n',
]))

out_path = r"c:\Users\diyal\OneDrive\Desktop\research\kaggle_vaani_training.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created: {out_path}")
