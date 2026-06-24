"""Abstract base class for TTS models (Template Method pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseTTSModel(nn.Module, ABC):
    """Contract every TTS architecture in this repo must satisfy."""

    @abstractmethod
    def compute_loss(self, *, text_ids: torch.Tensor, text_mask: torch.Tensor,
                     audio_codes: torch.Tensor, audio_mask: torch.Tensor,
                     speaker_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return a dict with at least ``{"loss": scalar}`` plus metrics."""

    @abstractmethod
    @torch.no_grad()
    def generate(self, *, text_ids: torch.Tensor, speaker_index: torch.Tensor,
                 max_frames: int, **sampling) -> torch.Tensor:
        """Autoregressively sample Mimi codes ``[B, Q, frames]``."""

    # ---- shared utilities -------------------------------------------- #
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save_pretrained(self, path: str, *, extra: Optional[Dict] = None) -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {"state_dict": self.state_dict(), "extra": extra or {}}
        torch.save(payload, path)

    def load_pretrained(self, path: str, *, strict: bool = True,
                        map_location="cpu") -> Dict:
        payload = torch.load(path, map_location=map_location)
        sd = payload.get("state_dict", payload)
        self.load_state_dict(sd, strict=strict)
        return payload.get("extra", {})
