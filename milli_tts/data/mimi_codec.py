"""Frozen Mimi neural audio codec wrapper.

Mimi (Kyutai) is a streaming neural audio codec operating at 24 kHz with a
12.5 Hz frame rate and residual vector quantization (RVQ). We use it purely as
a **frozen** tokenizer/detokenizer:

    waveform [1, T]  --encode-->  codes [Q, frames]  (integers in [0, 2048))
    codes [Q, frames] --decode-->  waveform [1, T']

The TTS language model never touches raw audio — it only predicts Mimi codes,
which the frozen Mimi decoder turns back into a 24 kHz waveform. Freezing Mimi
is what keeps training cheap and inference fast.

The wrapper hides the ``moshi`` loader behind a clean interface and is exposed
as a process-wide singleton through :class:`StaticMemoryCache`. A lightweight
``dummy`` backend is provided so the rest of the pipeline can be smoke-tested
on a machine without the (heavy) moshi/torch-cuda stack.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from milli_tts.core.logger import get_logger
from milli_tts.core.static_memory_cache import StaticMemoryCache

log = get_logger("data.mimi")


class MimiCodec(nn.Module):
    """Thin, frozen wrapper around the Mimi codec."""

    def __init__(self, model: nn.Module, *, sample_rate: int, frame_rate: float,
                 num_codebooks: int, codebook_size: int, backend: str = "moshi") -> None:
        super().__init__()
        self.model = model
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.backend = backend
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, *, allow_dummy: Optional[bool] = None,
                    device: Optional[torch.device] = None) -> "MimiCodec":
        """Build (and cache) the codec described by ``config.json``.

        ``allow_dummy`` controls the fallback when the real Mimi weights can't be
        loaded. ``None`` (the default) reads ``codec.allow_dummy`` from config —
        ``False`` for real training, so a load failure raises instead of silently
        training on garbage dummy codes. Pass ``True`` explicitly for CPU smoke
        tests where ``moshi`` isn't installed.
        """

        def _factory() -> "MimiCodec":
            cfg = StaticMemoryCache.config().codec
            dev = device or StaticMemoryCache.device()
            permit_dummy = (getattr(cfg, "allow_dummy", False)
                            if allow_dummy is None else allow_dummy)
            try:
                model = cls._load_moshi_mimi(cfg.hf_repo, cfg.num_codebooks, dev)
                codec = cls(
                    model,
                    sample_rate=cfg.sample_rate,
                    frame_rate=cfg.frame_rate,
                    num_codebooks=cfg.num_codebooks,
                    codebook_size=cfg.codebook_size,
                    backend="moshi",
                )
                log.info("Loaded Mimi codec (%s) on %s", cfg.hf_repo, dev)
            except Exception as exc:  # pragma: no cover - depends on env
                if not permit_dummy:
                    raise RuntimeError(
                        "Could not load the real Mimi codec and "
                        "codec.allow_dummy=False, so training refuses to fall back "
                        "to the pseudo-random DUMMY codec (it never converges). "
                        "Install the real codec: `pip install moshi "
                        "huggingface_hub` (and verify CUDA works — check "
                        "`torch.cuda.get_arch_list()`)."
                        f"{cls._gpu_arch_hint(exc)} For CPU smoke tests set "
                        f"codec.allow_dummy=true. Cause: {exc}") from exc
                log.warning(
                    "Could not load real Mimi (%s). Falling back to DUMMY codec "
                    "for smoke testing. Install `moshi` for real audio.", exc)
                codec = cls(
                    _DummyMimi(cfg.num_codebooks, cfg.codebook_size,
                               cfg.sample_rate, cfg.frame_rate),
                    sample_rate=cfg.sample_rate,
                    frame_rate=cfg.frame_rate,
                    num_codebooks=cfg.num_codebooks,
                    codebook_size=cfg.codebook_size,
                    backend="dummy",
                )
            return codec.to(dev)

        return StaticMemoryCache.get_or_create("codec::mimi", _factory)

    @staticmethod
    def _gpu_arch_hint(exc: Exception) -> str:
        """Extra remedy text when the failure is a GPU compute-arch mismatch.

        A "no kernel image is available for execution on the device" error means
        the installed PyTorch wheel has no kernels for this GPU's compute
        capability — classically a Blackwell GPU (sm_120: RTX 50-series, RTX PRO
        6000, B200) on a cu124 wheel that only ships up to sm_90. Point straight
        at the cu128 fix instead of the generic "check CUDA" advice.
        """
        msg = str(exc).lower()
        if "no kernel image" not in msg and "kernel image is available" not in msg:
            return ""
        try:  # best-effort; never let diagnostics raise
            import torch
            arches = ", ".join(torch.cuda.get_arch_list()) or "<none>"
            cc = torch.cuda.get_device_capability(0)
            gpu = f" (this GPU is sm_{cc[0]}{cc[1]}; wheel built for: {arches})"
        except Exception:
            gpu = ""
        return (
            f" This looks like a GPU compute-arch mismatch{gpu}: the installed "
            "PyTorch has no kernels for your GPU. Blackwell GPUs (sm_120) need a "
            "cu128 wheel — reinstall with: `pip install --index-url "
            "https://download.pytorch.org/whl/cu128 \"torch>=2.8,<2.10\" "
            "\"torchaudio>=2.8,<2.10\"` (driver R570+ required).")

    @staticmethod
    def _load_moshi_mimi(hf_repo: str, num_codebooks: int,
                         device: torch.device) -> nn.Module:
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        cfg = StaticMemoryCache.config()
        token = cfg.huggingface.token
        mimi_name = getattr(loaders, "MIMI_NAME",
                            "tokenizer-e351c8d8-checkpoint125.safetensors")
        default_repo = getattr(loaders, "DEFAULT_REPO",
                               "kyutai/moshiko-pytorch-bf16")
        # The moshi-format Mimi weight (MIMI_NAME) ships in the moshi LM repo,
        # NOT in the standalone `kyutai/mimi` repo (which is transformers-format
        # `model.safetensors`). Try the moshi repo first, then the configured
        # repo, so a stale `codec.hf_repo` in config.json can't 404 us.
        candidates, seen = [], set()
        for repo in (default_repo, hf_repo):
            if repo and repo not in seen:
                seen.add(repo)
                candidates.append(repo)

        last_err: Exception | None = None
        for repo in candidates:
            try:
                weight_path = hf_hub_download(repo, mimi_name, token=token)
                mimi = loaders.get_mimi(weight_path, device=device)
                mimi.set_num_codebooks(num_codebooks)
                log.info("Mimi weights loaded from %s/%s", repo, mimi_name)
                return mimi
            except Exception as exc:  # try next candidate repo
                log.debug("Mimi load from %s failed: %s", repo, exc)
                last_err = exc
        raise last_err if last_err else RuntimeError("Mimi load failed")

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """waveform -> integer codes.

        Args:
            wav: ``[B, 1, T]`` or ``[1, T]`` or ``[T]`` float waveform at
                ``self.sample_rate``.
        Returns:
            ``[B, Q, frames]`` long tensor of codcode indices.
        """
        wav = self._ensure_batched(wav).to(next(self.model.parameters()).device
                                            if any(True for _ in self.model.parameters())
                                            else wav.device)
        codes = self.model.encode(wav)  # [B, Q, frames]
        return codes.long()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """integer codes -> waveform ``[B, 1, T]``."""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        wav = self.model.decode(codes)
        return wav

    @staticmethod
    def _ensure_batched(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)
        return wav.to(torch.float32)

    def frames_for_seconds(self, seconds: float) -> int:
        return int(round(seconds * self.frame_rate))


class _DummyMimi(nn.Module):
    """Deterministic stand-in for Mimi used when ``moshi`` is unavailable.

    It is *not* a real codec — encode/decode are pseudo-inverse hashes — but it
    has the right tensor shapes so the data + training + inference plumbing can
    be exercised end-to-end on CPU.
    """

    def __init__(self, num_codebooks: int, codebook_size: int,
                 sample_rate: int, frame_rate: float) -> None:
        super().__init__()
        self.q = num_codebooks
        self.cb = codebook_size
        self.sr = sample_rate
        self.fr = frame_rate
        self.samples_per_frame = int(round(sample_rate / frame_rate))

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        b, _, t = wav.shape
        frames = max(1, t // self.samples_per_frame)
        chunks = wav[..., : frames * self.samples_per_frame]
        chunks = chunks.reshape(b, frames, self.samples_per_frame)
        feats = chunks.mean(dim=-1)  # [b, frames]
        codes = []
        for q in range(self.q):
            idx = ((feats * (q + 3) * 1000.0).long().abs() % self.cb)
            codes.append(idx)
        return torch.stack(codes, dim=1)  # [b, q, frames]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        b, q, frames = codes.shape
        val = (codes.float() / self.cb).mean(dim=1)  # [b, frames]
        wav = val.repeat_interleave(self.samples_per_frame, dim=1)
        return (wav * 2 - 1).unsqueeze(1)  # [b, 1, T]
