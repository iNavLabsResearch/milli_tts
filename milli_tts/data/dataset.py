"""IndicVoices dataset adapter.

The ai4bharat/IndicVoices corpus is an STT corpus: each row has an audio clip,
its transcript (``normalized`` / ``text``), a ``speaker_id`` (~400 distinct
speakers), ``gender``, ``lang`` and rich metadata. For TTS we simply invert the
task — **text is the input, the spoken waveform is the target** — which is
exactly what a delayed-streams TTS model trains on. So yes: STT labels are
perfectly good TTS training pairs.

This class yields lightweight per-utterance samples:

    {
        "text_ids":      LongTensor[L],     # tokenized transcript
        "wav":           FloatTensor[T],    # 24 kHz mono waveform (target)
        "speaker_index": int,               # VoiceBank index for the speaker
        "speaker_id":    str,
        "gender": str, "lang": str, "duration": float,
    }

Heavy Mimi encoding is intentionally deferred to the trainer (done on GPU in
batches), so this dataset stays cheap and works inside DataLoader workers.
Supports HF streaming mode (no full download — ideal for Colab) and a cached
map-style mode.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from milli_tts.core.logger import get_logger
from milli_tts.core.static_memory_cache import StaticMemoryCache
from milli_tts.data.text_tokenizer import TextTokenizer
from milli_tts.data.voice_bank import VoiceBank
from milli_tts.utils.audio import resample

log = get_logger("data.dataset")

# Candidate column names (IndicVoices variants differ slightly across configs).
_TEXT_FIELDS = ("normalized", "text", "verbatim", "sentence")
_SPEAKER_FIELDS = ("speaker_id", "speaker", "client_id")


class IndicVoicesDataset(IterableDataset):
    def __init__(self, *, split: Optional[str] = None,
                 tokenizer: Optional[TextTokenizer] = None,
                 voice_bank: Optional[VoiceBank] = None,
                 register_voices: bool = True) -> None:
        super().__init__()
        cfg = StaticMemoryCache.config()
        self.cfg = cfg
        self.hf = cfg.huggingface
        self.split = split or self.hf.dataset_split
        self.target_sr = cfg.codec.sample_rate
        self.min_sec = cfg.training.min_audio_seconds
        self.max_sec = cfg.training.max_audio_seconds
        self.register_voices = register_voices
        self.tokenizer = tokenizer or TextTokenizer.from_config()
        self.voice_bank = voice_bank or VoiceBank.from_config()
        self._hf_dataset = None

    # ------------------------------------------------------------------ #
    def _build_hf_dataset(self):
        from datasets import Audio, load_dataset

        log.info("Loading %s [config=%s split=%s streaming=%s]",
                 self.hf.dataset_repo, self.hf.dataset_config, self.split,
                 self.hf.streaming)
        kwargs = dict(split=self.split, streaming=self.hf.streaming,
                      token=self.hf.token)
        if self.hf.dataset_config:
            try:
                ds = load_dataset(self.hf.dataset_repo, self.hf.dataset_config,
                                  **kwargs)
            except Exception:
                ds = load_dataset(self.hf.dataset_repo, **kwargs)
        else:
            ds = load_dataset(self.hf.dataset_repo, **kwargs)
        # Ensure the audio column decodes to the codec sample rate.
        audio_col = self._find_audio_col(ds)
        if audio_col:
            ds = ds.cast_column(audio_col, Audio(sampling_rate=self.target_sr))
        return ds

    @staticmethod
    def _find_audio_col(ds) -> Optional[str]:
        features = getattr(ds, "features", None)
        if features:
            for name in ("audio", "audio_filepath", "wav"):
                if name in features:
                    return name
        return "audio"

    @staticmethod
    def _pick(row: Dict, fields) -> Optional[str]:
        for f in fields:
            v = row.get(f)
            if isinstance(v, str) and v.strip():
                return v
        return None

    # ------------------------------------------------------------------ #
    def _extract_wav(self, row: Dict) -> Optional[torch.Tensor]:
        audio = row.get("audio") or row.get("wav") or row.get("audio_filepath")
        if isinstance(audio, dict) and "array" in audio:
            arr = torch.as_tensor(audio["array"], dtype=torch.float32)
            sr = int(audio.get("sampling_rate", self.target_sr))
            if arr.dim() > 1:
                arr = arr.mean(dim=-1)
            if sr != self.target_sr:
                arr = resample(arr.unsqueeze(0), sr, self.target_sr).squeeze(0)
            return arr
        return None

    def _process_row(self, row: Dict) -> Optional[Dict]:
        text = self._pick(row, _TEXT_FIELDS)
        if not text:
            return None
        wav = self._extract_wav(row)
        if wav is None or wav.numel() == 0:
            return None
        dur = wav.numel() / self.target_sr
        if dur < self.min_sec or dur > self.max_sec:
            return None

        speaker_id = self._pick(row, _SPEAKER_FIELDS) or "unknown"
        gender = row.get("gender")
        lang = row.get("lang") or self.hf.dataset_config
        if self.register_voices:
            spk_index = self.voice_bank.add_or_get(speaker_id, gender=gender,
                                                   lang=lang)
        else:
            spk_index = (self.voice_bank.index_of(speaker_id)
                         if speaker_id in self.voice_bank else 0)

        text_ids = self.tokenizer.encode_tensor(text)
        return {
            "text_ids": text_ids,
            "wav": wav,
            "speaker_index": spk_index,
            "speaker_id": speaker_id,
            "gender": gender or "",
            "lang": lang or "",
            "duration": float(dur),
        }

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[Dict]:
        if self._hf_dataset is None:
            self._hf_dataset = self._build_hf_dataset()
        # Shard across DataLoader workers when streaming.
        worker = torch.utils.data.get_worker_info()
        ds = self._hf_dataset
        if worker is not None and self.hf.streaming:
            ds = ds.shard(num_shards=worker.num_workers, index=worker.id) \
                if hasattr(ds, "shard") else ds
        for i, row in enumerate(ds):
            try:
                sample = self._process_row(row)
            except Exception as exc:  # robust to occasional bad rows
                log.debug("Skipping row %d: %s", i, exc)
                continue
            if sample is not None:
                yield sample
