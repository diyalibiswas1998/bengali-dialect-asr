"""Masked multi-task CTC, dialect, and routing losses."""

import torch
import torch.nn.functional as F


def load_balancing_loss(gate_probs: torch.Tensor | None, topk_indices: torch.Tensor | None) -> torch.Tensor:
    if gate_probs is None or topk_indices is None:
        device = gate_probs.device if gate_probs is not None else "cpu"
        return torch.zeros((), device=device)
    num_experts = gate_probs.shape[-1]
    importance = gate_probs.float().mean(dim=0)
    assignment = F.one_hot(topk_indices, num_classes=num_experts).float().mean(dim=(0, 1))
    return num_experts * torch.sum(importance * assignment)


def multitask_loss(outputs, batch, ctc_weight=1.0, dialect_weight=0.2, balance_weight=0.01):
    log_probs = outputs["logits"].float().log_softmax(-1).transpose(0, 1)
    ctc = F.ctc_loss(
        log_probs,
        batch["targets"],
        outputs["output_lengths"],
        batch["target_lengths"],
        blank=0,
        zero_infinity=True,
    )
    label_mask = batch["dialect_label_mask"].bool()
    if dialect_weight and label_mask.any():
        head_loss = F.cross_entropy(outputs["dialect_logits"][label_mask], batch["dialect_labels"][label_mask])
        if outputs.get("gate_probs") is not None:
            router_loss = F.nll_loss(
                outputs["gate_probs"][label_mask].clamp_min(1e-9).log(),
                batch["dialect_labels"][label_mask],
            )
            dialect = 0.5 * (head_loss + router_loss)
        else:
            dialect = head_loss
    else:
        dialect = ctc.new_zeros(())
    balance = load_balancing_loss(outputs.get("gate_probs"), outputs.get("topk_indices")).to(ctc.device)
    total = ctc_weight * ctc + dialect_weight * dialect + balance_weight * balance
    return total, {"ctc": ctc.detach(), "dialect": dialect.detach(), "balance": balance.detach()}
