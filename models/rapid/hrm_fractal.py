"""Adapted ("rapid") nested HRM variant with explicit demonstration conditioning (paper, Appendix A). Model C.
"""
from dataclasses import dataclass
import torch
from torch import nn
from models.rapid.hrm_baseline import TaskConditionedInputNetwork, TaskConditionedOutputHead, StackedBlock, init_module_

@dataclass
class FractalTaskConfig:
    vocab_size: int = 11
    canvas: int = 30
    dim: int = 256
    num_heads: int = 4
    expansion: int = 4
    num_layers_L: int = 4
    num_layers_H: int = 4
    N: int = 3
    T: int = 4
    N_inner: int = 2
    T_inner: int = 2

class FractalTaskRapid(nn.Module):

    def __init__(self, cfg: FractalTaskConfig):
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
        z_L_outer = z0.clone()
        z_L_inner = z0.clone()
        return (z_H, z_L_outer, z_L_inner)

    def _fractal_L_step(self, z_L_outer, z_L_inner, z_H, x_tilde, track_grad_last: bool):
        cfg = self.cfg
        total_inner = cfg.N_inner * cfg.T_inner
        if not track_grad_last:
            with torch.no_grad():
                for j in range(total_inner):
                    z_L_inner = self.f_L(z_L_inner + z_L_outer + z_H + x_tilde)
                    if (j + 1) % cfg.T_inner == 0:
                        z_L_outer = self.f_L(z_L_outer + z_L_inner)
            return (z_L_outer, z_L_inner)
        with torch.no_grad():
            for j in range(total_inner - 1):
                z_L_inner = self.f_L(z_L_inner + z_L_outer + z_H + x_tilde)
                if (j + 1) % cfg.T_inner == 0:
                    z_L_outer = self.f_L(z_L_outer + z_L_inner)
        z_L_inner = self.f_L(z_L_inner + z_L_outer + z_H + x_tilde)
        z_L_outer = self.f_L(z_L_outer + z_L_inner)
        return (z_L_outer, z_L_inner)

    def forward(self, query_canvas, demo_inputs, demo_outputs, demo_mask, z_H, z_L_outer, z_L_inner):
        cfg = self.cfg
        x_tilde = self.f_I(query_canvas, demo_inputs, demo_outputs, demo_mask)
        total_steps = cfg.N * cfg.T
        with torch.no_grad():
            for i in range(total_steps - 1):
                z_L_outer, z_L_inner = self._fractal_L_step(z_L_outer, z_L_inner, z_H, x_tilde, track_grad_last=False)
                if (i + 1) % cfg.T == 0:
                    z_H = self.f_H(z_H + z_L_outer)
        z_L_outer, z_L_inner = self._fractal_L_step(z_L_outer, z_L_inner, z_H, x_tilde, track_grad_last=True)
        z_H = self.f_H(z_H + z_L_outer)
        y_hat = self.f_O(z_H)
        return (z_H, z_L_outer, z_L_inner, y_hat)
