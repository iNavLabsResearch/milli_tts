"""Reusable transformer building blocks: RMSNorm, RoPE, attention, MLP, block.

A clean, dependency-light decoder-only stack with:
* Rotary position embeddings (RoPE)
* RMSNorm pre-normalization
* Optional KV cache for fast autoregressive inference
* Causal masking

These blocks are shared by both the temporal **backbone** transformer (over
audio frames) and the small **depth** transformer (over codebooks within a
frame), so the two only differ in size/config, not code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [S, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, H, S, Dh]; cos/sin: [S, Dh]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor
    length: int = 0


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                *, causal: bool = True, cache: Optional[KVCache] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """``key_padding_mask``: optional ``[B, S_kv]`` bool, ``True`` = real key,
        ``False`` = padding (never attended to). It is combined with the causal
        mask so a query never sees padded *keys*. When ``None`` the original
        ``is_causal`` fast paths run unchanged (so the KV-cache inference path is
        untouched)."""
        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            prev_len = cache.length
            if prev_len == 0:
                cache.k = k
                cache.v = v
            else:
                cache.k = torch.cat([cache.k, k], dim=2)
                cache.v = torch.cat([cache.v, v], dim=2)
            k, v = cache.k, cache.v
            cache.length = k.shape[2]
            total = cache.length
            # Causality must match training (whole sequence causal). With a KV
            # cache the q-tokens are the newest `s`; they may attend to all
            # cached keys (always in their past) plus be causal *among
            # themselves*. We build an explicit block-causal mask covering both
            # the single-token decode (s==1) and the multi-token prefill cases.
            if key_padding_mask is None and s == 1:
                attn = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=False)
            else:
                # [s, total] bool: True = keep. New token i (abs pos prev_len+i)
                # may see key j iff j <= prev_len + i.
                qpos = torch.arange(prev_len, prev_len + s, device=x.device)
                kpos = torch.arange(total, device=x.device)
                keep = (kpos.unsqueeze(0) <= qpos.unsqueeze(1))  # [s, total]
                keep = keep.unsqueeze(0).unsqueeze(0)            # [1, 1, s, total]
                keep = self._apply_key_padding(keep, key_padding_mask, total)
                attn = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=keep, dropout_p=0.0)
        elif key_padding_mask is None:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal)
        else:
            # No cache: fold the (optional) causal mask and the key-padding mask
            # into one boolean attn_mask. SDPA forbids passing both `attn_mask`
            # and `is_causal`, so the causal part lives in the mask here.
            keep = key_padding_mask[:, None, None, :].bool()  # [B, 1, 1, S]
            keep = keep.expand(b, 1, s, s)
            if causal:
                causal_keep = torch.ones(s, s, dtype=torch.bool,
                                         device=x.device).tril()
                keep = keep & causal_keep
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=keep,
                dropout_p=self.dropout if self.training else 0.0)

        out = attn.transpose(1, 2).contiguous().view(b, s, -1)
        return self.proj(out), cache

    @staticmethod
    def _apply_key_padding(keep: torch.Tensor,
                           key_padding_mask: Optional[torch.Tensor],
                           total: int) -> torch.Tensor:
        """AND a ``[1, 1, S, total]`` causal-keep mask with an optional
        ``[B, S_kv]`` key-padding mask (sliced to the cached length)."""
        if key_padding_mask is None:
            return keep
        kpm = key_padding_mask[:, :total].bool()  # [B, total]
        return keep & kpm[:, None, None, :]


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, ffn_mult)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                *, causal: bool = True, cache: Optional[KVCache] = None,
                key_padding_mask: Optional[torch.Tensor] = None):
        h, cache = self.attn(self.norm1(x), cos, sin, causal=causal,
                             cache=cache, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, cache


class TransformerStack(nn.Module):
    """A stack of pre-norm transformer blocks with a final norm."""

    def __init__(self, *, dim: int, depth: int, heads: int, ffn_mult: int,
                 rope_theta: float, dropout: float = 0.0,
                 grad_checkpoint: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.rope_theta = rope_theta
        self.grad_checkpoint = grad_checkpoint
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, ffn_mult, dropout) for _ in range(depth)])
        self.norm = RMSNorm(dim)

    def rope(self, seq_len: int, device, dtype, offset: int = 0):
        cos, sin = build_rope_cache(seq_len + offset, self.head_dim,
                                    self.rope_theta, device, dtype)
        return cos[offset:offset + seq_len], sin[offset:offset + seq_len]

    def forward(self, x: torch.Tensor, *, causal: bool = True,
                caches: Optional[list] = None, pos_offset: int = 0,
                key_padding_mask: Optional[torch.Tensor] = None):
        seq_len = x.shape[1]
        cos, sin = self.rope(seq_len, x.device, x.dtype, offset=pos_offset)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            if self.grad_checkpoint and self.training and cache is None:
                x, cache = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, causal=causal,
                    key_padding_mask=key_padding_mask, use_reentrant=False)
            else:
                x, cache = block(x, cos, sin, causal=causal, cache=cache,
                                 key_padding_mask=key_padding_mask)
            new_caches.append(cache)
        x = self.norm(x)
        return x, new_caches
