"""TTSInferenceEngine — voice_id + text -> 24 kHz waveform.

Loads the trained checkpoint, the VoiceBank (voice_id -> speaker index) and the
frozen Mimi decoder, then synthesizes speech. Designed for the T4 latency
target:

* text prefix is prefilled into the backbone KV cache once;
* each 12.5 Hz frame is one cheap backbone step + Q tiny depth steps;
* fp16 + KV cache + (optional) torch.compile keep per-frame latency low, so the
  *time-to-first-audio-chunk* can be pushed under ~100 ms while the rest streams
  faster than real time.

Two ways to specify a voice (Strategy, mirrored from training):
* a known ``voice_id`` from the VoiceBank (embedding-table speaker), or
* ``reference_audio`` of an unseen speaker for zero-shot prefix cloning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch

from milli_tts.core.config import AppConfig
from milli_tts.core.logger import get_logger
from milli_tts.core.static_memory_cache import StaticMemoryCache
from milli_tts.data.mimi_codec import MimiCodec
from milli_tts.data.text_tokenizer import TextTokenizer
from milli_tts.data.voice_bank import VoiceBank
from milli_tts.models.factory import build_model
from milli_tts.training.checkpoint import CheckpointManager
from milli_tts.utils.audio import load_audio, save_audio

log = get_logger("inference.engine")


@dataclass
class SynthesisResult:
    wav: torch.Tensor          # [1, T] float @ sample_rate
    sample_rate: int
    voice_id: str
    text: str
    num_frames: int
    latency_ms: float          # total generation wall-time
    first_chunk_ms: float      # time-to-first-audio-chunk (the 100ms target)
    real_time_factor: float    # audio_seconds / wall_seconds (>1 = faster than RT)

    def save(self, path: str) -> str:
        save_audio(path, self.wav, self.sample_rate)
        return path


class TTSInferenceEngine:
    def __init__(self, cfg: Optional[AppConfig] = None,
                 checkpoint: Optional[str] = None) -> None:
        self.cfg = cfg or StaticMemoryCache.config()
        self.icfg = self.cfg.inference
        self.device = StaticMemoryCache.device(self.icfg.device)

        self.tokenizer = TextTokenizer.from_config()
        self.voice_bank = VoiceBank.from_config()
        self.codec = MimiCodec.from_config(device=self.device)

        self.model = build_model(
            text_vocab_size=self.tokenizer.vocab_size,
            num_codebooks=self.cfg.codec.num_codebooks,
            codebook_size=self.cfg.codec.codebook_size,
        ).to(self.device)

        self._load_checkpoint(checkpoint)
        self.model.eval()
        self._maybe_half()
        log.info("Inference engine ready on %s | %d voices | %.1fM params",
                 self.device, len(self.voice_bank),
                 self.model.num_parameters(False) / 1e6)

    # ------------------------------------------------------------------ #
    def _load_checkpoint(self, checkpoint: Optional[str]) -> None:
        ckpt = CheckpointManager(self.cfg.paths.checkpoint_dir)
        path = checkpoint or ckpt.resolve("best") or ckpt.resolve("latest")
        if not path:
            log.warning("No checkpoint found in %s; using randomly-initialized "
                        "weights (smoke-test only).", self.cfg.paths.checkpoint_dir)
            return
        ckpt.load(path, model=self.model, map_location=self.device)

    def _maybe_half(self) -> None:
        if self.device.type == "cuda" and self.icfg.precision.lower() == "fp16":
            self.model.half()

    # ------------------------------------------------------------------ #
    def list_voices(self) -> List[str]:
        return self.voice_bank.list_voice_ids()

    def register_reference(self, voice_id: str, reference_audio: str) -> None:
        """Encode a reference clip and store it as a cloneable prefix voice."""
        wav, _ = load_audio(reference_audio, target_sr=self.cfg.codec.sample_rate)
        max_s = self.cfg.voice.reference_clip_seconds
        wav = wav[..., : int(max_s * self.cfg.codec.sample_rate)]
        codes = self.codec.encode(wav.to(self.device))[0]  # [Q, frames]
        self.voice_bank.save_prefix(voice_id, codes)
        log.info("Registered reference voice '%s' (%d frames).",
                 voice_id, codes.shape[-1])

    # ------------------------------------------------------------------ #
    def _resolve_voice(self, voice_id: Optional[str]):
        """Return (speaker_index_tensor, prefix_codes_or_None)."""
        vid = voice_id or self.cfg.voice.default_voice_id
        prefix = self.voice_bank.load_prefix(vid)
        if vid in self.voice_bank:
            idx = self.voice_bank.index_of(vid)
        else:
            if prefix is None:
                log.warning("voice_id '%s' unknown and no reference prefix; "
                            "using default speaker index 0.", vid)
            idx = 0
        spk = torch.tensor([idx], dtype=torch.long, device=self.device)
        return spk, prefix

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def synthesize(self, text: str, voice_id: Optional[str] = None, *,
                   temperature: Optional[float] = None,
                   top_k: Optional[int] = None, top_p: Optional[float] = None,
                   max_seconds: Optional[float] = None) -> SynthesisResult:
        spk, prefix = self._resolve_voice(voice_id)
        text_ids = self.tokenizer.encode_tensor(text).unsqueeze(0).to(self.device)

        max_frames = self.codec.frames_for_seconds(
            max_seconds or self.icfg.max_gen_seconds)
        temperature = self.icfg.temperature if temperature is None else temperature
        top_k = self.icfg.top_k if top_k is None else top_k
        top_p = self.icfg.top_p if top_p is None else top_p

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.time()

        # measure time-to-first-chunk by generating a tiny first segment
        first_chunk_frames = max(1, self.icfg.stream_chunk_frames)
        first_codes = self.model.generate(
            text_ids=text_ids, speaker_index=spk, max_frames=first_chunk_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            prefix_codes=prefix, eos_silence_frames=10 ** 9)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        first_chunk_ms = (time.time() - t_start) * 1000.0

        # full generation (fresh, simple + correct; prefill dominates anyway)
        codes = self.model.generate(
            text_ids=text_ids, speaker_index=spk, max_frames=max_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            prefix_codes=prefix)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.time() - t_start) * 1000.0

        wav = self.codec.decode(codes)[0] if codes.shape[-1] > 0 \
            else torch.zeros(1, 1, device=self.device)
        audio_seconds = wav.shape[-1] / self.cfg.codec.sample_rate
        rtf = audio_seconds / max(latency_ms / 1000.0, 1e-6)

        return SynthesisResult(
            wav=wav.float().cpu(), sample_rate=self.cfg.codec.sample_rate,
            voice_id=voice_id or self.cfg.voice.default_voice_id, text=text,
            num_frames=int(codes.shape[-1]), latency_ms=latency_ms,
            first_chunk_ms=first_chunk_ms, real_time_factor=rtf)
