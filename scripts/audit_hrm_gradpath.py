"""Gradient-path audit for HRM-F C1 and C2 on a small deterministic operator: checks which lower-stack applications receive gradients.
"""
import copy
from types import SimpleNamespace

import torch
from torch import nn

from models.hrm_fractal_c1 import (
    HierarchicalReasoningModel_ACTV1_Inner as C1Inner,
)
from models.hrm_fractal_c2 import (
    HierarchicalReasoningModel_ACTV1_Inner as C2Inner,
)


class ToyL(nn.Module):
    """Tiny deterministic shared operator used only to audit gradient flow."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states, input_injection, **kwargs):
        return torch.tanh(
            self.linear(hidden_states + input_injection)
        )


def make_dummy(module):
    return SimpleNamespace(
        config=SimpleNamespace(
            N_inner=2,
            T_inner=2,
        ),
        L_level=module,
    )


torch.manual_seed(1234)

B, S, D = 2, 5, 8

base_L = ToyL(D)
L_c1 = copy.deepcopy(base_L)
L_c2 = copy.deepcopy(base_L)

c1 = make_dummy(L_c1)
c2 = make_dummy(L_c2)

z_M = torch.randn(B, S, D)
z_L = torch.randn(B, S, D)
z_H = torch.randn(B, S, D)
inp = torch.randn(B, S, D)

# ------------------------------------------------------------
# Forward: must be numerically identical
# ------------------------------------------------------------

m1, l1 = C1Inner._fractal_L_step(
    c1,
    z_M.clone(),
    z_L.clone(),
    z_H.clone(),
    inp.clone(),
    {},
    track_grad_last=True,
)

m2, l2 = C2Inner._fractal_L_step(
    c2,
    z_M.clone(),
    z_L.clone(),
    z_H.clone(),
    inp.clone(),
    {},
    track_grad_last=True,
)

max_diff_M = (m1.detach() - m2.detach()).abs().max().item()
max_diff_L = (l1.detach() - l2.detach()).abs().max().item()

print("=== FORWARD AUDIT ===")
print(f"max |z_M C1-C2| = {max_diff_M:.12g}")
print(f"max |z_L C1-C2| = {max_diff_L:.12g}")

if not torch.allclose(m1.detach(), m2.detach(), atol=1e-7, rtol=1e-7):
    raise RuntimeError("FAIL: C1 and C2 z_M forward differ")

if not torch.allclose(l1.detach(), l2.detach(), atol=1e-7, rtol=1e-7):
    raise RuntimeError("FAIL: C1 and C2 z_L forward differ")

# ------------------------------------------------------------
# Backward: gradients should differ
# ------------------------------------------------------------

loss1 = (m1.square().mean() + l1.square().mean())
loss2 = (m2.square().mean() + l2.square().mean())

loss1.backward()
loss2.backward()

g1 = L_c1.linear.weight.grad
g2 = L_c2.linear.weight.grad

norm1 = g1.norm().item()
norm2 = g2.norm().item()
grad_diff = (g1 - g2).norm().item()

print()
print("=== GRADIENT AUDIT ===")
print(f"C1 grad norm       = {norm1:.12g}")
print(f"C2 grad norm       = {norm2:.12g}")
print(f"||grad_C1-grad_C2|| = {grad_diff:.12g}")

if not torch.isfinite(g1).all():
    raise RuntimeError("FAIL: non-finite C1 gradients")

if not torch.isfinite(g2).all():
    raise RuntimeError("FAIL: non-finite C2 gradients")

if grad_diff <= 1e-10:
    raise RuntimeError("FAIL: gradient paths appear identical")

print()
print("PASS: same forward, different finite gradient path.")
print("C2 changes credit assignment without changing inference computation.")
