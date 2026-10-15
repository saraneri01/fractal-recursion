"""Adapted ("rapid") TRM baseline with explicit demonstration conditioning (paper, Appendix A). Model A.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.rapid_trm.layers import Attention, SwiGLU, RotaryEmbedding, rms_norm
from models.components.rapid_trm.common import trunc_normal_init_

@dataclass
class TRMTaskConfig:
    vocab_size: int = 11
    canvas: int = 30
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    L_layers: int = 2
    H_cycles: int = 3
    L_cycles: int = 6
    rms_norm_eps: float = 1e-05
    rope_theta: float = 10000.0

    @property
    def seq_len(self) -> int:
        return self.canvas * self.canvas + 1

class TRMBlock(nn.Module):

    def __init__(self, cfg: TRMTaskConfig):
        super().__init__()
        self.attn = Attention(cfg.hidden_size, cfg.hidden_size // cfg.num_heads, cfg.num_heads, cfg.num_heads, causal=False)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.expansion)
        self.eps = cfg.rms_norm_eps

    def forward(self, cos_sin, hidden_states):
        hidden_states = rms_norm(hidden_states + self.attn(cos_sin=cos_sin, hidden_states=hidden_states), self.eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), self.eps)
        return hidden_states

class ReasoningModule(nn.Module):

    def __init__(self, cfg: TRMTaskConfig, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([TRMBlock(cfg) for _ in range(num_layers)])

    def forward(self, hidden_states, input_injection, cos_sin):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states
from models.components.rapid_trm.demo_encoder import DemoContextEncoder

class TRMTaskInputNetwork(nn.Module):

    def __init__(self, cfg: TRMTaskConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.demo_encoder = DemoContextEncoder(cfg.vocab_size, cfg.hidden_size)
        nn.init.trunc_normal_(self.token_emb.weight, std=1.0, a=-2.0, b=2.0)

    def forward(self, query_canvas, demo_inputs, demo_outputs, demo_mask):
        B = query_canvas.shape[0]
        query_tokens = self.token_emb(query_canvas.view(B, -1))
        context = self.demo_encoder(demo_inputs, demo_outputs, demo_mask).unsqueeze(1)
        return torch.cat([context, query_tokens], dim=1)

class TRMTaskOutputHead(nn.Module):

    def __init__(self, cfg: TRMTaskConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=1.0 / math.sqrt(cfg.hidden_size), a=-2.0, b=2.0)

    def forward(self, z_H):
        return self.proj(z_H[:, 1:, :])

class TRMTaskBaseline(nn.Module):

    def __init__(self, cfg: TRMTaskConfig):
        super().__init__()
        self.cfg = cfg
        self.f_I = TRMTaskInputNetwork(cfg)
        self.L_level = ReasoningModule(cfg, cfg.L_layers)
        self.f_O = TRMTaskOutputHead(cfg)
        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_emb = RotaryEmbedding(dim=head_dim, max_position_embeddings=cfg.seq_len, base=cfg.rope_theta)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(cfg.hidden_size), std=1.0), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(cfg.hidden_size), std=1.0), persistent=True)

    def init_state(self, batch_size, device):
        z_H = self.H_init.to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, self.cfg.seq_len, -1).clone()
        z_L = self.L_init.to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, self.cfg.seq_len, -1).clone()
        return (z_H, z_L)

    def forward(self, query_canvas, demo_inputs, demo_outputs, demo_mask, z_H, z_L):
        cfg = self.cfg
        x_tilde = self.f_I(query_canvas, demo_inputs, demo_outputs, demo_mask)
        cos_sin = self.rotary_emb()
        with torch.no_grad():
            for _ in range(cfg.H_cycles - 1):
                for _ in range(cfg.L_cycles):
                    z_L = self.L_level(z_L, z_H + x_tilde, cos_sin)
                z_H = self.L_level(z_H, z_L, cos_sin)
        for _ in range(cfg.L_cycles):
            z_L = self.L_level(z_L, z_H + x_tilde, cos_sin)
        z_H = self.L_level(z_H, z_L, cos_sin)
        y_hat = self.f_O(z_H)
        return (z_H, z_L, y_hat)

def count_params(module: nn.Module) -> int:
    return sum((p.numel() for p in module.parameters()))
