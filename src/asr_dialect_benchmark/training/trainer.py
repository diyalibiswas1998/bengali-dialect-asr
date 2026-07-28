import os
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..common.utils import ensure_dir
from ..data.dataset import BengaliDialectDataset, collate_batch
from ..losses.ctc_losses import LoadBalancingLoss
from ..modeling.asr_model import BengaliDialectASR


class Trainer:
    def __init__(self, cfg, tokenizer=None):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.writer = SummaryWriter(log_dir=cfg.training.log_dir)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.load_balancing_criterion = LoadBalancingLoss(importance_weight=cfg.loss.load_balancing_weight)
        self.scaler = GradScaler(enabled=cfg.training.use_amp and torch.cuda.is_available())
        self.best_val_loss = float("inf")
        self.best_checkpoint_path = None
        self.step = 0
        self.model = BengaliDialectASR(cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.training.lr)
        self.train_dataset = BengaliDialectDataset(cfg.data.train_manifest, tokenizer=tokenizer)
        self.val_dataset = BengaliDialectDataset(cfg.data.val_manifest, tokenizer=tokenizer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, collate_fn=lambda batch: collate_batch(batch, tokenizer=tokenizer))
        self.val_loader = DataLoader(self.val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=lambda batch: collate_batch(batch, tokenizer=tokenizer))
        ensure_dir(cfg.training.output_dir)

    def _step_batch(self, batch):
        audio = batch["audio"].to(self.device)
        dialect_labels = batch["dialect_label"].to(self.device)
        target = batch["target"].to(self.device)
        audio_length = batch["audio_length"].to(self.device)
        target_length = batch["target_length"].to(self.device)
        with autocast(enabled=self.scaler.is_enabled()):
            outputs = self.model(audio, attention_mask=None, labels=target, dialect_labels=dialect_labels)
            logits = outputs["logits"]
            dialect_logits = outputs["dialect_logits"]
            ctc_logits = logits.transpose(0, 1)
            input_lengths = self.model.encoder._get_feat_extract_output_lengths(audio_length)
            input_lengths = torch.clamp(input_lengths, max=ctc_logits.size(0))
            ctc_loss = self.model.loss_fn(ctc_logits.log_softmax(-1), target, input_lengths, target_length)
            class_loss = self.criterion(dialect_logits, dialect_labels)
            total_loss = class_loss + ctc_loss
            if self.cfg.loss.use_dialect_loss:
                total_loss = total_loss + class_loss
            if self.cfg.loss.use_load_balancing and self.cfg.model.use_router:
                gate_probs = torch.softmax(dialect_logits, dim=-1)
                topk_vals, topk_indices = self.model.moe.topk_gating(gate_probs, self.cfg.model.top_k if hasattr(self.cfg.model, 'top_k') else 2)
                total_loss = total_loss + self.load_balancing_criterion(gate_probs, topk_indices)
            if self.cfg.training.stage == 1:
                total_loss = class_loss
        return total_loss, class_loss, ctc_loss

    def train(self):
        self.model.train()
        for epoch in range(self.cfg.training.max_epochs):
            pbar = tqdm(total=len(self.train_loader), desc=f"epoch {epoch}")
            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad(set_to_none=True)
                loss, class_loss, ctc_loss = self._step_batch(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.writer.add_scalar("train/loss", loss.item(), self.step)
                self.writer.add_scalar("train/class_loss", class_loss.item(), self.step)
                self.writer.add_scalar("train/ctc_loss", ctc_loss.item(), self.step)
                self.step += 1
                pbar.update(1)
            pbar.close()
            val_loss = self.validate(epoch)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_checkpoint_path = str(Path(self.cfg.training.output_dir) / f"checkpoint_epoch_{epoch}.pt")
                torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, self.best_checkpoint_path)
                self.writer.add_scalar("val/best_val_loss", val_loss, epoch)
            self.writer.flush()

    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                loss, _, _ = self._step_batch(batch)
                total_loss += loss.item()
        avg_loss = total_loss / max(1, len(self.val_loader))
        self.writer.add_scalar("val/loss", avg_loss, epoch)
        return avg_loss

    def save(self, path: str):
        ensure_dir(str(Path(path).parent))
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, path)

    def inference(self, batch):
        self.model.eval()
        with torch.no_grad():
            audio = batch["audio"].to(self.device)
            outputs = self.model(audio)
            return outputs
