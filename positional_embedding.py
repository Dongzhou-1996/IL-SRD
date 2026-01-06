import torch
import torch.nn as nn
import math


class PositionEmbeddingSine(nn.Module):
    def __init__(self, hidden_dim: int):

        super().__init__()
        self.hidden_dim = hidden_dim
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                             (-math.log(10000.0) / hidden_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        pe = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        pe[:, 0::2] = torch.sin(torch.tensor([1.0], device=x.device) * self.div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(torch.tensor([1.0], device=x.device) * self.div_term)  # 奇数维度

        return pe