import torch
import torch.nn as nn


class DialectRouter(nn.Module):
    """Predicts dialect probabilities over four dialects."""

    def __init__(self, input_dim: int, num_dialects: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_dialects = num_dialects
        self.proj1 = nn.Linear(input_dim, max(64, input_dim // 2))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj2 = nn.Linear(max(64, input_dim // 2), num_dialects)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            x = hidden_states.mean(dim=1)
        else:
            x = hidden_states
        x = self.proj1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.proj2(x)
        return torch.softmax(x, dim=-1)
