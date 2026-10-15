"""Adapted ("rapid") nested TRM variant with explicit demonstration conditioning (paper, Appendix A). Model C.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from models.components.rapid_trm.layers import RotaryEmbedding
from models.components.rapid_trm.common import trunc_normal_init_
from models.rapid.trm_baseline import ReasoningModule, TRMTaskInputNetwork, TRMTaskOutputHead

@dataclass
class TRMFTaskConfig:
    vocab_size: int = 11
    canvas: int = 30
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    L_layers: int = 2
    H_cycles: int = 3
    M_cycles: int = 2
    L_cycles: int = 3
    rms_norm_eps: float = 1e-05
    rope_theta: float = 10000.0

    @property
    def seq_len(self) -> int:
        return self.canvas * self.canvas + 1

class TRMFTaskModel(nn.Module):

    def __init__(self, cfg: TRMFTaskConfig):
        super().__init__()
        self.cfg = cfg
        self.f_I = TRMTaskInputNetwork(cfg)
        self.L_level = ReasoningModule(cfg, cfg.L_layers)
        self.f_O = TRMTaskOutputHead(cfg)
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_emb = RotaryEmbedding(dim=head_dim, max_position_embeddings=cfg.seq_len, base=cfg.rope_theta)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(cfg.hidden_size), std=1.0), persistent=True)
        self.M_init = nn.Buffer(trunc_normal_init_(torch.empty(cfg.hidden_size), std=1.0), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(cfg.hidden_size), std=1.0), persistent=True)

    def init_state(self, batch_size, device):

        def expand(buf):
            return buf.to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, self.cfg.seq_len, -1).clone()
        return (expand(self.H_init), expand(self.M_init), expand(self.L_init))

    def _inner_cycle(self, z_M, z_L, z_H_ctx, x_tilde, cos_sin, track_grad_last: bool):
        cfg = self.cfg
        if not track_grad_last:
            with torch.no_grad():
                for _ in range(cfg.L_cycles):
                    z_L = self.L_level(z_L, z_M + x_tilde, cos_sin)
                z_M = self.L_level(z_M, z_L, cos_sin)
            return (z_M, z_L)
        with torch.no_grad():
            for _ in range(cfg.L_cycles - 1):
                z_L = self.L_level(z_L, z_M + x_tilde, cos_sin)
        z_L = self.L_level(z_L, z_M + x_tilde, cos_sin)
        z_M = self.L_level(z_M, z_L, cos_sin)
        return (z_M, z_L)

    def forward(self, query_canvas, demo_inputs, demo_outputs, demo_mask, z_H, z_M, z_L):
        cfg = self.cfg
        x_tilde = self.f_I(query_canvas, demo_inputs, demo_outputs, demo_mask)
        cos_sin = self.rotary_emb()
        with torch.no_grad():
            for _ in range(cfg.H_cycles - 1):
                for _ in range(cfg.M_cycles):
                    z_M, z_L = self._inner_cycle(z_M, z_L, z_H, x_tilde, cos_sin, track_grad_last=False)
                z_H = self.L_level(z_H, z_M, cos_sin)
        for _ in range(cfg.M_cycles - 1):
            z_M, z_L = self._inner_cycle(z_M, z_L, z_H, x_tilde, cos_sin, track_grad_last=False)
        z_M, z_L = self._inner_cycle(z_M, z_L, z_H, x_tilde, cos_sin, track_grad_last=True)
        z_H = self.L_level(z_H, z_M, cos_sin)
        y_hat = self.f_O(z_H)
        return (z_H, z_M, z_L, y_hat)

def count_params(module: nn.Module) -> int:
    return sum((p.numel() for p in module.parameters()))
