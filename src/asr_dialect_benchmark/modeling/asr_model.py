import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from .moe import SparseMixtureOfExperts


class BengaliDialectASR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Read num_dialects from config so it works for both 4-dialect and 11-district setups
        num_dialects = getattr(cfg.model, "num_dialects", 4)
        num_tokens = getattr(cfg.model, "num_tokens", 32)

        self.encoder = Wav2Vec2Model(Wav2Vec2Config(
            vocab_size=num_tokens,
            hidden_size=768,          # smaller hidden size for faster training
            num_hidden_layers=6,      # reduced layers for research experiments
            num_attention_heads=12,
            feat_proj_dropout=0.0,
            hidden_dropout=0.1,
            layerdrop=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        ))
        self.encoder.feature_extractor._freeze_parameters()
        self.moe = SparseMixtureOfExperts(
            hidden_size=self.encoder.config.hidden_size,
            num_dialects=num_dialects,
            top_k=getattr(cfg.model, "top_k", 2),
            dropout=cfg.model.dropout,
            use_router=cfg.model.use_router,
            use_shared_expert=cfg.model.use_shared_expert,
        )
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_dialects)
        self.ctc_head = nn.Linear(self.encoder.config.hidden_size, num_tokens)
        self.loss_fn = nn.CTCLoss(blank=getattr(cfg.model, "blank_index", 0), zero_infinity=True)

    def forward(self, input_values, attention_mask=None, labels=None, dialect_labels=None):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        moep = self.moe(hidden_states)
        logits = self.ctc_head(moep)
        dialect_logits = self.classifier(moep.mean(dim=1))
        return {"logits": logits, "dialect_logits": dialect_logits, "hidden_states": moep}
