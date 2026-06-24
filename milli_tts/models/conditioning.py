"""Voice conditioning strategies (Strategy pattern).

A ``voice_id`` is turned into a conditioning vector that biases generation
toward a particular speaker. Two interchangeable strategies:

* :class:`EmbeddingTableConditioner` — a learned embedding per known speaker
  index (the "voice_id" pattern used by ElevenLabs/Sarvam-style catalogs).
  Best quality for speakers seen during training.

* :class:`PrefixConditioner` — returns a neutral/zero bias and relies on the
  model being primed with a reference-clip prefix at inference time
  (zero-shot voice cloning for unseen speakers).

Both expose the same ``forward(speaker_index) -> [B, D]`` interface so the
model is agnostic to which is configured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class VoiceConditioner(nn.Module, ABC):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward(self, speaker_index: torch.Tensor) -> torch.Tensor:
        """speaker_index [B] long -> conditioning bias [B, d_model]."""
        raise NotImplementedError


class EmbeddingTableConditioner(VoiceConditioner):
    def __init__(self, d_model: int, max_speakers: int,
                 embedding_dim: int) -> None:
        super().__init__(d_model)
        self.table = nn.Embedding(max_speakers + 1, embedding_dim)
        self.proj = (nn.Identity() if embedding_dim == d_model
                     else nn.Linear(embedding_dim, d_model, bias=False))
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, speaker_index: torch.Tensor) -> torch.Tensor:
        return self.proj(self.table(speaker_index))


class PrefixConditioner(VoiceConditioner):
    """No learned per-speaker bias; identity conditioning (zero-shot via prefix)."""

    def __init__(self, d_model: int) -> None:
        super().__init__(d_model)
        self.null = nn.Parameter(torch.zeros(d_model))

    def forward(self, speaker_index: torch.Tensor) -> torch.Tensor:
        b = speaker_index.shape[0]
        return self.null.unsqueeze(0).expand(b, -1)


def build_conditioner(strategy: str, *, d_model: int, max_speakers: int,
                      embedding_dim: int) -> VoiceConditioner:
    strategy = (strategy or "embedding_table").lower()
    if strategy in ("embedding_table", "embedding", "table"):
        return EmbeddingTableConditioner(d_model, max_speakers, embedding_dim)
    if strategy in ("prefix", "zero_shot", "clone"):
        return PrefixConditioner(d_model)
    raise ValueError(f"Unknown voice conditioning strategy: {strategy}")
