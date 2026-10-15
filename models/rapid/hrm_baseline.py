"""Adapted ("rapid") HRM baseline with explicit demonstration conditioning (paper, Appendix A). Model A.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.rapid.layers import init_module_
from models.components.rapid.hrm_core import StackedBlock

class DemoContextEncoder(nn.Module):

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        mid1 = 32
        mid2 = 64
        self.grid_encoder = nn.Sequential(nn.Conv2d(vocab_size, mid1, kernel_size=3, padding=1, bias=False), nn.GELU(), nn.Conv2d(mid1, mid2, kernel_size=3, stride=2, padding=1, bias=False), nn.GELU(), nn.Conv2d(mid2, mid2, kernel_size=3, stride=2, padding=1, bias=False), nn.GELU())
        self.grid_proj = nn.Linear(mid2 * 8 * 8, dim, bias=False)
        self.input_role = nn.Parameter(torch.zeros(dim))
        self.output_role = nn.Parameter(torch.zeros(dim))
        self.pair_proj = nn.Linear(2 * dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        nn.init.normal_(self.input_role, std=0.02)
        nn.init.normal_(self.output_role, std=0.02)

    def _encode_grids(self, grids: torch.Tensor) -> torch.Tensor:
        B, D, H, W = grids.shape
        x = F.one_hot(grids.clamp(0, self.vocab_size - 1), num_classes=self.vocab_size).float()
        x = x.permute(0, 1, 4, 2, 3).reshape(B * D, self.vocab_size, H, W)
        z = self.grid_encoder(x)
        z = z.flatten(1)
        z = self.grid_proj(z)
        return z.reshape(B, D, self.dim)

    def forward(self, demo_inputs, demo_outputs, demo_mask):
        z_input = self._encode_grids(demo_inputs) + self.input_role
        z_output = self._encode_grids(demo_outputs) + self.output_role
        pair_repr = self.pair_proj(torch.cat([z_input, z_output], dim=-1))
        mask = demo_mask.to(pair_repr.dtype).unsqueeze(-1)
        summed = (pair_repr * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        context = summed / denominator
        return self.norm(context)

class TaskConditionedInputNetwork(nn.Module):

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.demo_encoder = DemoContextEncoder(vocab_size, dim)
        nn.init.trunc_normal_(self.token_emb.weight, std=1.0, a=-2.0, b=2.0)

    def forward(self, query_canvas, demo_inputs, demo_outputs, demo_mask):
        B = query_canvas.shape[0]
        query_tokens = self.token_emb(query_canvas.view(B, -1))
        context = self.demo_encoder(demo_inputs, demo_outputs, demo_mask).unsqueeze(1)
        return torch.cat([context, query_tokens], dim=1)

class TaskConditionedOutputHead(nn.Module):

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=1.0 / math.sqrt(dim), a=-2.0, b=2.0)

    def forward(self, z_H):
        grid_states = z_H[:, 1:, :]
        return self.proj(grid_states)

@dataclass
class FlatTaskConfig:
    vocab_size: int = 11
    canvas: int = 30
    dim: int = 256
    num_heads: int = 4
    expansion: int = 4
    num_layers_L: int = 4
    num_layers_H: int = 4
    N: int = 3
    T: int = 4

class FlatTaskRapid(nn.Module):

    def __init__(self, cfg: FlatTaskConfig):
        super().__init__()
        self.cfg = cfg
        self.f_I = TaskConditionedInputNetwork(cfg.vocab_size, cfg.dim)
        self.f_L = StackedBlock(cfg.dim, cfg.num_heads, cfg.expansion, cfg.num_layers_L)
        self.f_H = StackedBlock(cfg.dim, cfg.num_heads, cfg.expansion, cfg.num_layers_H)
        self.f_O = TaskConditionedOutputHead(cfg.vocab_size, cfg.dim)
        init_module_(self.f_L)
        init_module_(self.f_H)

    def init_state(self, batch_size: int, device):
        seq_len = self.cfg.canvas * self.cfg.canvas + 1
        z0 = torch.empty(batch_size, seq_len, self.cfg.dim, device=device)
        nn.init.trunc_normal_(z0, std=1.0, a=-2.0, b=2.0)
        z_H = z0.clone()
        z_L = z0.clone()
        return (z_H, z_L)

    def forward(self, query_canvas, demo_inputs, demo_outputs, demo_mask, z_H, z_L):
        cfg = self.cfg
        x_tilde = self.f_I(query_canvas, demo_inputs, demo_outputs, demo_mask)
        total_steps = cfg.N * cfg.T
        with torch.no_grad():
            for i in range(total_steps - 1):
                z_L = self.f_L(z_L + z_H + x_tilde)
                if (i + 1) % cfg.T == 0:
                    z_H = self.f_H(z_H + z_L)
        z_L = self.f_L(z_L + z_H + x_tilde)
        z_H = self.f_H(z_H + z_L)
        y_hat = self.f_O(z_H)
        return (z_H, z_L, y_hat)

def count_params(module: nn.Module) -> int:
    return sum((p.numel() for p in module.parameters()))
