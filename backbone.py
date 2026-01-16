import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class backbone(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.LayerNorm(hidden_dim)  # 增强特征稳定性
        )


    def forward(self, state):
        x = self.fc(state)
        return x

