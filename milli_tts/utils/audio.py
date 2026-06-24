"""Audio I/O and resampling helpers built on torchaudio with a soundfile fallback."""

from __future__ import annotations

import os
from typing import Tuple

import torch


def load_audio(path: str, target_sr: int | None = None,
               mono: bool = True) -> Tuple[torch.Tensor, int]:
    """Load an audio file -> (waveform[C, T] float32, sample_rate).

    Falls back to soundfile if torchaudio's backend can't read the file.
    """
    wav: torch.Tensor
    sr: int
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)
    except Exception:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.ascontiguousarray(data.T))  # [C, T]

    wav = wav.to(torch.float32)
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    # Prefer torchaudio, but fall back to librosa so a broken/absent torchaudio
    # binary (a common Colab torch/torchaudio version mismatch) is non-fatal.
    try:
        import torchaudio

        return torchaudio.functional.resample(wav, orig_sr, target_sr)
    except Exception:
        import librosa
        import numpy as np

        arr = wav.detach().cpu().numpy()
        out = librosa.resample(np.ascontiguousarray(arr), orig_sr=orig_sr,
                               target_sr=target_sr, axis=-1)
        return torch.from_numpy(out).to(wav.dtype)


def save_audio(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    """Save a [C, T] or [T] float waveform to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.detach().to(torch.float32).clamp_(-1.0, 1.0).cpu()
    try:
        import torchaudio

        torchaudio.save(path, wav, sample_rate)
    except Exception:
        import soundfile as sf

        sf.write(path, wav.squeeze(0).numpy(), sample_rate)
