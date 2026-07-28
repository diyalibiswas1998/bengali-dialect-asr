import torch
import torch.nn as nn

from .experts import DialectExpert, SharedExpert
from .router import DialectRouter


class SparseMixtureOfExperts(nn.Module):
    def __init__(self, hidden_size: int, num_dialects: int = 4, top_k: int = 2, dropout: float = 0.1, use_router: bool = True, use_shared_expert: bool = True):
        super().__init__()
        self.use_router = use_router
        self.use_shared_expert = use_shared_expert
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.router = DialectRouter(hidden_size, num_dialects=num_dialects, dropout=dropout) if use_router else None
        self.shared_expert = SharedExpert(hidden_size=hidden_size, dropout=dropout) if use_shared_expert else None
        self.dialect_experts = nn.ModuleList([DialectExpert(hidden_size=hidden_size, dropout=dropout) for _ in range(num_dialects)])

    def topk_gating(self, gate_probs: torch.Tensor, top_k: int):
        return torch.topk(gate_probs, k=min(top_k, gate_probs.size(-1)), dim=-1)

    def forward(self, hidden_states: torch.Tensor, return_aux: bool = False):
        pooled = hidden_states.mean(dim=1)
        if self.use_router:
            gate_probs = self.router(pooled)
        else:
            gate_probs = torch.full((pooled.size(0), self.dialect_experts.__len__()), 1.0 / self.dialect_experts.__len__(), device=pooled.device)
        topk_vals, topk_indices = self.topk_gating(gate_probs, self.top_k)
        expert_outputs = []
        for expert_idx in range(self.top_k):
            expert_ids = topk_indices[:, expert_idx]
            per_batch = []
            for sample_idx, expert_id in enumerate(expert_ids.tolist()):
                per_batch.append(self.dialect_experts[expert_id](pooled[sample_idx]))
            per_batch = torch.stack(per_batch, dim=0)
            expert_outputs.append(per_batch * topk_vals[:, expert_idx].unsqueeze(-1))
        fusion = torch.zeros_like(pooled)
        if self.use_shared_expert:
            fusion = fusion + self.shared_expert(pooled)
        for output in expert_outputs:
            fusion = fusion + output
        fused_sequence = hidden_states + fusion.unsqueeze(1).expand_as(hidden_states)
        if return_aux:
            return fused_sequence, gate_probs, topk_indices
        return fused_sequence
