"""MMS-300M CTC baseline and dialect-aware MoE model."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from .moe import SparseMixtureOfExperts, masked_mean


def _value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


class BengaliDialectASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = _value(config, "model", config)
        self.pretrained_model = _value(model_config, "pretrained_model", "facebook/mms-300m")
        self.num_dialects = int(_value(model_config, "num_dialects", 4))
        self.use_moe = bool(_value(model_config, "use_moe", True))
        self.encoder = AutoModel.from_pretrained(self.pretrained_model)
        if bool(_value(model_config, "gradient_checkpointing", True)):
            self.encoder.gradient_checkpointing_enable()
        hidden_size = int(self.encoder.config.hidden_size)
        if self.use_moe:
            self.moe = SparseMixtureOfExperts(
                hidden_size=hidden_size,
                num_dialects=self.num_dialects,
                top_k=int(_value(model_config, "top_k", 2)),
                dropout=float(_value(model_config, "dropout", 0.1)),
                use_router=bool(_value(model_config, "use_router", True)),
                use_shared_expert=bool(_value(model_config, "use_shared_expert", True)),
            )
        else:
            self.moe = None
        self.dialect_classifier = nn.Linear(hidden_size, self.num_dialects)
        self.ctc_head = nn.Linear(hidden_size, int(_value(model_config, "num_tokens", 64)))

    def feature_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder._get_feat_extract_output_lengths(input_lengths).to(torch.long)

    def set_phase(self, phase: int, top_layers: int = 4) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if phase >= 2:
            layers = self.encoder.encoder.layers
            for layer in layers[-top_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            # The final normalization is part of the top representation.
            if hasattr(self.encoder.encoder, "layer_norm"):
                for parameter in self.encoder.encoder.layer_norm.parameters():
                    parameter.requires_grad = True

    def forward(self, input_values=None, attention_mask=None, input_lengths=None, routing_inputs=None):
        if routing_inputs is not None:
            if self.moe is None:
                return {"gate_probs": None, "topk_indices": None}
            gate_probs, _, topk_indices = self.moe.route(routing_inputs)
            return {"gate_probs": gate_probs, "topk_indices": topk_indices}
        encoded = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden_states = encoded.last_hidden_state
        if input_lengths is None:
            input_lengths = attention_mask.sum(-1) if attention_mask is not None else input_values.new_full((input_values.shape[0],), input_values.shape[1], dtype=torch.long)
        output_lengths = self.feature_lengths(input_lengths)
        output_lengths = output_lengths.clamp(max=hidden_states.shape[1])
        time = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None, :]
        feature_mask = time < output_lengths[:, None]
        if self.moe is not None:
            hidden_states, gate_probs, topk_indices, router_input = self.moe(hidden_states, feature_mask)
        else:
            gate_probs, topk_indices, router_input = None, None, None
        pooled = masked_mean(hidden_states, feature_mask)
        return {
            "logits": self.ctc_head(hidden_states),
            "dialect_logits": self.dialect_classifier(pooled),
            "gate_probs": gate_probs,
            "topk_indices": topk_indices,
            "router_input": router_input,
            "output_lengths": output_lengths,
        }
