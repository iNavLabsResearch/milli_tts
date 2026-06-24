"""Token sampling utilities (temperature / top-k / top-p / greedy)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_logits(logits: torch.Tensor, *, temperature: float = 1.0,
                  top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Sample one token id per row. logits: [B, V] -> [B] long."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-5)

    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        remove = cumprobs - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
