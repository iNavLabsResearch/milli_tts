"""Batch collation for the delayed-streams TTS model.

Pads variable-length transcripts and waveforms into a dense batch. Mimi
encoding and the delay/interleave pattern are applied *after* collation (on
GPU, in the trainer / model) so the collator stays cheap and worker-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class TTSBatch:
    text_ids: torch.Tensor       # [B, Lt] long, padded with text_pad_id
    text_mask: torch.Tensor      # [B, Lt] bool, True where real token
    wav: torch.Tensor            # [B, Tw] float, padded with 0
    wav_lengths: torch.Tensor    # [B] long, true sample counts
    speaker_index: torch.Tensor  # [B] long
    speaker_ids: List[str]
    langs: List[str]

    def to(self, device) -> "TTSBatch":
        return TTSBatch(
            text_ids=self.text_ids.to(device, non_blocking=True),
            text_mask=self.text_mask.to(device, non_blocking=True),
            wav=self.wav.to(device, non_blocking=True),
            wav_lengths=self.wav_lengths.to(device, non_blocking=True),
            speaker_index=self.speaker_index.to(device, non_blocking=True),
            speaker_ids=self.speaker_ids,
            langs=self.langs,
        )

    def __len__(self) -> int:
        return self.text_ids.shape[0]


class DelayedStreamCollator:
    def __init__(self, text_pad_id: int) -> None:
        self.text_pad_id = text_pad_id

    def __call__(self, samples: List[Dict]) -> TTSBatch:
        samples = [s for s in samples if s is not None]
        b = len(samples)
        lt = max(s["text_ids"].numel() for s in samples)
        tw = max(s["wav"].numel() for s in samples)

        text_ids = torch.full((b, lt), self.text_pad_id, dtype=torch.long)
        text_mask = torch.zeros((b, lt), dtype=torch.bool)
        wav = torch.zeros((b, tw), dtype=torch.float32)
        wav_lengths = torch.zeros(b, dtype=torch.long)
        speaker_index = torch.zeros(b, dtype=torch.long)
        speaker_ids: List[str] = []
        langs: List[str] = []

        for i, s in enumerate(samples):
            t = s["text_ids"]
            text_ids[i, : t.numel()] = t
            text_mask[i, : t.numel()] = True
            w = s["wav"]
            wav[i, : w.numel()] = w
            wav_lengths[i] = w.numel()
            speaker_index[i] = int(s["speaker_index"])
            speaker_ids.append(s["speaker_id"])
            langs.append(s.get("lang", ""))

        return TTSBatch(text_ids, text_mask, wav, wav_lengths,
                        speaker_index, speaker_ids, langs)
