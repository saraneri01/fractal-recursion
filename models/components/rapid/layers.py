from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, max_seq_len: int=1024, base: float=10000.0):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        return (self.cos[:seq_len].to(device=device, dtype=dtype), self.sin[:seq_len].to(device=device, dtype=dtype))

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q2 = q * cos + _rotate_half(q) * sin
    k2 = k * cos + _rotate_half(k) * sin
    return (q2, k2)

class SelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        cos, sin = self.rope(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)

class SwiGLU(nn.Module):

    def __init__(self, dim: int, expansion: int=4):
        super().__init__()
        hidden = int(dim * expansion * 2 / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, expansion: int=4):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads)
        self.mlp = SwiGLU(dim, expansion)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x

def truncated_lecun_normal_(tensor: torch.Tensor, fan_in: int):
    std = 1.0 / math.sqrt(fan_in)
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)

def init_module_(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            truncated_lecun_normal_(m.weight, fan_in=m.in_features)
