import torch
import torch.nn as nn


class LoadBalancingLoss(nn.Module):
    def __init__(self, importance_weight: float = 0.01):
        super().__init__()
        self.importance_weight = importance_weight

    def forward(self, gate_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        if gate_probs.ndim != 2:
            raise ValueError("gate_probs must have shape [batch, num_experts]")
        expert_counts = torch.bincount(topk_indices.flatten(), minlength=gate_probs.size(1)).to(gate_probs.device)
        mean_probs = gate_probs.mean(dim=0)
        target_probs = torch.full_like(mean_probs, fill_value=1.0 / gate_probs.size(1))
        loss = torch.mean((expert_counts.float() / max(1, expert_counts.sum())) - mean_probs) + torch.mean((mean_probs - target_probs) ** 2)
        return self.importance_weight * loss.abs()
