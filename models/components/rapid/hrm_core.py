from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from models.components.rapid.layers import TransformerBlock, init_module_

class StackedBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, expansion: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, expansion) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
