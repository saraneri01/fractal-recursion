from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DemoContextEncoder(nn.Module):
    """
    Compress the ARC demonstrations into one context token.
    This encoder is identical in Model A and Model C.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        mid1 = 32
        mid2 = 64
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(vocab_size, mid1, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid1, mid2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid2, mid2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
        )
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
