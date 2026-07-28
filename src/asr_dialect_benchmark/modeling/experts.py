import torch
import torch.nn as nn


class ResidualFeedForward(nn.Module):
    """Lightweight residual feed-forward expert block."""

    def __init__(self, hidden_size: int, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ff_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(residual + x)


class SharedExpert(nn.Module):
    def __init__(self, hidden_size: int, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.block = ResidualFeedForward(hidden_size, ff_dim=ff_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DialectExpert(nn.Module):
    def __init__(self, hidden_size: int, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.block = ResidualFeedForward(hidden_size, ff_dim=ff_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
