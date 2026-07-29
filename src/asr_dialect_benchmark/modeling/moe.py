"""Utterance-routed sparse dialect mixture of experts."""

import torch
import torch.nn as nn

from .experts import DialectExpert, SharedExpert
from .router import DialectRouter


def masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden_states.mean(dim=1)
    weights = mask.to(hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class SparseMixtureOfExperts(nn.Module):
    def __init__(self, hidden_size: int, num_dialects: int = 4, top_k: int = 2, dropout: float = 0.1, use_router: bool = True, use_shared_expert: bool = True):
        super().__init__()
        self.use_router = use_router
        self.use_shared_expert = use_shared_expert
        self.top_k = min(top_k, num_dialects)
        self.router = DialectRouter(hidden_size, num_dialects=num_dialects, dropout=dropout) if use_router else None
        self.shared_expert = SharedExpert(hidden_size=hidden_size, dropout=dropout) if use_shared_expert else None
        self.dialect_experts = nn.ModuleList(DialectExpert(hidden_size=hidden_size, dropout=dropout) for _ in range(num_dialects))

    def route(self, pooled: torch.Tensor):
        if self.router is None:
            gate_probs = pooled.new_full((pooled.shape[0], len(self.dialect_experts)), 1.0 / len(self.dialect_experts))
        else:
            gate_probs = self.router(pooled)
        topk_values, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return gate_probs, topk_values, topk_indices

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        pooled = masked_mean(hidden_states, attention_mask)
        gate_probs, topk_values, topk_indices = self.route(pooled)
        fusion = torch.zeros_like(pooled)
        # Dispatch only selected samples to each expert. Top-1 therefore
        # performs half the dialect-expert work of top-2.
        for expert_id, expert in enumerate(self.dialect_experts):
            sample_indices, slots = torch.where(topk_indices == expert_id)
            if sample_indices.numel() == 0:
                continue
            expert_output = expert(pooled.index_select(0, sample_indices))
            weighted = expert_output * topk_values[sample_indices, slots].unsqueeze(-1)
            fusion = fusion.index_add(0, sample_indices, weighted)
        if self.shared_expert is not None:
            fusion = fusion + self.shared_expert(pooled)
        return hidden_states + fusion.unsqueeze(1), gate_probs, topk_indices, pooled
