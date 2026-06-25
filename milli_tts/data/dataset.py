"""Hugging Face TTS dataset adapter (IndicTTS-Hindi / IndicVoices).

Primary target: **SPRINGLab/IndicTTS-Hindi** — a clean studio TTS corpus whose
rows carry only ``audio``, ``text`` (Devanagari Hindi) and ``gender`` (a
``ClassLabel`` of ``female``/``male``). It has **no per-row speaker_id or lang**,
so this adapter derives a stable speaker id from gender (``hi_female`` /
``hi_male`` → two learned voice embeddings) and assumes the corpus language.

It also stays backward-compatible with **ai4bharat/IndicVoices**, an STT corpus
(audio + ``normalized``/``text`` transcript + ``speaker_id`` + ``gender`` +
``lang``). For TTS we simply invert the task — **text is the input, the spoken
waveform is the target** — so STT labels are perfectly good TTS training pairs.

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
from milli_tts.utils.audio import load_audio, resample

log = get_logger("data.dataset")

# Candidate column names (IndicVoices variants differ slightly across configs).
_TEXT_FIELDS = ("normalized", "text", "verbatim", "sentence")
_SPEAKER_FIELDS = ("speaker_id", "speaker", "client_id")

# IndicVoices is organised as one HF *config per language*. Map the short
# language code (what each row's `lang` field carries, e.g. "hi") to that
# config name so a user can pick a language by code from config.json.
_LANG_CODE_TO_CONFIG = {
    "as": "assamese", "bn": "bengali", "brx": "bodo", "doi": "dogri",
    "gu": "gujarati", "hi": "hindi", "kn": "kannada", "ks": "kashmiri",
    "kok": "konkani", "mai": "maithili", "ml": "malayalam", "mni": "manipuri",
    "mr": "marathi", "ne": "nepali", "or": "odia", "pa": "punjabi",
    "sa": "sanskrit", "sat": "santali", "sd": "sindhi", "ta": "tamil",
    "te": "telugu", "ur": "urdu",
}
_CONFIG_TO_LANG_CODE = {v: k for k, v in _LANG_CODE_TO_CONFIG.items()}


def _normalize_langs(langs) -> set:
    """Turn a config `languages` list into a set of canonical lang codes.

    Returns an empty set to mean "keep everything" (no filter). Accepts codes
    ("hi"), config names ("hindi") or the wildcards "all"/"any"/"*".
    """
    if not langs:
        return set()
    out: set = set()
    for item in langs:
        s = str(item).strip().lower()
        if s in ("all", "any", "*", ""):
            return set()
        out.add(_CONFIG_TO_LANG_CODE.get(s, s))  # name -> code, else keep code
    return out


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
        self._gender_names = None  # filled from the gender ClassLabel feature

        # ---- language selection -------------------------------------- #
        # `languages` from config decides which language(s) to keep.
        self.allowed_langs = _normalize_langs(getattr(self.hf, "languages", None))
        self.dataset_config = self.hf.dataset_config

        # IndicVoices ships one HF *config per language*, so a single requested
        # language also auto-selects the matching `dataset_config`. This remap is
        # IndicVoices-specific — other corpora (e.g. SPRINGLab/IndicTTS-Hindi,
        # config "default") must keep their own config name, so gate it on repo.
        is_indicvoices = "indicvoices" in str(self.hf.dataset_repo).lower()
        if is_indicvoices and len(self.allowed_langs) == 1:
            only = next(iter(self.allowed_langs))
            mapped = _LANG_CODE_TO_CONFIG.get(only)
            if mapped and mapped != self.dataset_config:
                log.info("languages=['%s'] -> using IndicVoices config '%s' "
                         "(was '%s')", only, mapped, self.dataset_config)
                self.dataset_config = mapped

        # Fallback language for corpora without a per-row `lang` field (e.g.
        # IndicTTS-Hindi is entirely Hindi). Prefer the single requested lang,
        # else the lang implied by the config name, else "" (= unknown).
        if len(self.allowed_langs) == 1:
            self.default_lang = next(iter(self.allowed_langs))
        else:
            self.default_lang = _CONFIG_TO_LANG_CODE.get(
                str(self.dataset_config).lower(), "")

    # ------------------------------------------------------------------ #
    def _build_hf_dataset(self):
        import time as _time

        from datasets import Audio, load_dataset

        log.info("Loading %s [config=%s split=%s streaming=%s] keep_langs=%s — "
                 "this opens the stream (first shard fetch can take a minute)…",
                 self.hf.dataset_repo, self.dataset_config, self.split,
                 self.hf.streaming, sorted(self.allowed_langs) or "ALL")
        kwargs = dict(split=self.split, streaming=self.hf.streaming,
                      token=self.hf.token)
        t0 = _time.time()
        try:
            if self.dataset_config:
                ds = load_dataset(self.hf.dataset_repo, self.dataset_config,
                                  **kwargs)
            else:
                ds = load_dataset(self.hf.dataset_repo, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("gated", "403", "401", "authenticated",
                                       "access", "permission")):
                raise RuntimeError(
                    f"Cannot access {self.hf.dataset_repo}: it is GATED. Open "
                    f"https://huggingface.co/datasets/{self.hf.dataset_repo} "
                    f"while logged in as the token owner and click 'Agree and "
                    f"access', then retry. Original error: {exc}") from exc
            raise
        log.info("load_dataset() returned in %.1fs.", _time.time() - t0)
        # Decode audio OURSELVES (soundfile) instead of letting `datasets` use
        # torchcodec — torchcodec needs a matching FFmpeg/torch build that Colab
        # often lacks (libavutil.so.* missing). `decode=False` hands us the raw
        # encoded bytes; _extract_wav turns them into a waveform.
        audio_col = self._find_audio_col(ds)
        if audio_col:
            try:
                ds = ds.cast_column(audio_col, Audio(decode=False))
            except Exception as exc:
                log.warning("cast_column(decode=False) failed (%s); "
                            "leaving column as-is.", exc)
        # `gender` is often a HF ClassLabel, which streams as an int (0/1). Grab
        # its class names once so rows can be mapped back to "female"/"male".
        feats = getattr(ds, "features", None)
        gfeat = feats.get("gender") if feats else None
        self._gender_names = list(getattr(gfeat, "names", None) or []) or None
        log.info("Stream ready (audio col=%s, gender_names=%s, self-decode). "
                 "Pulling rows…", audio_col, self._gender_names)
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

    def _gender_name(self, gender) -> Optional[str]:
        """Normalize a row's gender to a lowercase name ("female"/"male").

        Handles both raw strings and HF ClassLabel ints (mapped via the class
        names captured at stream build time).
        """
        if gender is None or isinstance(gender, bool):
            return None
        if isinstance(gender, str):
            return gender.strip().lower() or None
        try:  # ClassLabel int -> name
            idx = int(gender)
        except (TypeError, ValueError):
            return None
        if self._gender_names and 0 <= idx < len(self._gender_names):
            return str(self._gender_names[idx]).strip().lower()
        return str(idx)

    # ------------------------------------------------------------------ #
    def _to_target(self, arr: torch.Tensor, sr: int) -> torch.Tensor:
        if arr.dim() > 1:                       # [C, T] or [T, C] -> mono [T]
            arr = arr.mean(dim=0 if arr.shape[0] < arr.shape[-1] else -1)
        if sr != self.target_sr:
            arr = resample(arr.unsqueeze(0), sr, self.target_sr).squeeze(0)
        return arr.to(torch.float32)

    def _decode_bytes(self, raw: bytes) -> Optional[torch.Tensor]:
        """Decode encoded audio bytes -> mono waveform at target sr (no torchcodec)."""
        import io

        try:
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            arr = torch.from_numpy(data.T)      # [C, T]
            return self._to_target(arr, int(sr))
        except Exception:
            try:                                # last resort for odd codecs
                import librosa

                data, sr = librosa.load(io.BytesIO(raw), sr=self.target_sr,
                                        mono=True)
                return torch.from_numpy(data).to(torch.float32)
            except Exception:
                return None

    def _extract_wav(self, row: Dict) -> Optional[torch.Tensor]:
        audio = row.get("audio") or row.get("wav") or row.get("audio_filepath")
        if isinstance(audio, dict):
            if "array" in audio and audio["array"] is not None:  # pre-decoded
                arr = torch.as_tensor(audio["array"], dtype=torch.float32)
                return self._to_target(arr, int(audio.get("sampling_rate",
                                                           self.target_sr)))
            if audio.get("bytes") is not None:                   # decode=False
                return self._decode_bytes(audio["bytes"])
            if audio.get("path"):                                # path on disk
                try:
                    wav, _ = load_audio(audio["path"], target_sr=self.target_sr)
                    return wav.squeeze(0)
                except Exception:
                    return None
        elif isinstance(audio, str):                             # bare path
            try:
                wav, _ = load_audio(audio, target_sr=self.target_sr)
                return wav.squeeze(0)
            except Exception:
                return None
        return None

    def _process_row(self, row: Dict) -> Optional[Dict]:
        sample, _ = self._process_row_diag(row)
        return sample

    def _process_row_diag(self, row: Dict):
        """Like _process_row but also returns a short skip reason for diagnostics."""
        text = self._pick(row, _TEXT_FIELDS)
        if not text:
            return None, "no_text"

        # Language filter FIRST (cheap, before decoding audio): drop rows whose
        # language isn't in the configured allow-list. Corpora without a per-row
        # `lang` (e.g. IndicTTS-Hindi) fall back to `default_lang`, which is the
        # requested language — so the whole single-language corpus passes.
        lang = row.get("lang") or self.default_lang
        lang_code = str(lang).strip().lower()
        lang_code = _CONFIG_TO_LANG_CODE.get(lang_code, lang_code)
        if self.allowed_langs and lang_code not in self.allowed_langs:
            return None, "wrong_lang"

        wav = self._extract_wav(row)
        if wav is None or wav.numel() == 0:
            return None, "no_audio"
        dur = wav.numel() / self.target_sr
        if dur < self.min_sec:
            return None, "too_short"
        if dur > self.max_sec:
            return None, "too_long"

        gender = self._gender_name(row.get("gender"))
        # Speaker identity: use an explicit speaker_id when the corpus has one;
        # otherwise derive a stable id from (lang, gender) — IndicTTS-Hindi then
        # yields exactly two voices ("hi_female"/"hi_male").
        speaker_id = self._pick(row, _SPEAKER_FIELDS)
        if not speaker_id:
            speaker_id = f"{lang_code or 'spk'}_{gender or 'unknown'}"
        if self.register_voices:
            spk_index = self.voice_bank.add_or_get(speaker_id, gender=gender,
                                                   lang=lang_code)
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
            "lang": lang_code or "",
            "duration": float(dur),
        }, "ok"

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[Dict]:
        if self._hf_dataset is None:
            self._hf_dataset = self._build_hf_dataset()
        # Shard across DataLoader workers when streaming so each worker sees a
        # DISJOINT slice (otherwise N workers would each replay the same rows).
        worker = torch.utils.data.get_worker_info()
        ds = self._hf_dataset
        stride, offset = 1, 0
        if worker is not None and self.hf.streaming and worker.num_workers > 1:
            if hasattr(ds, "shard"):
                ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
            else:  # fallback: stride the iterator by worker id (no duplication)
                stride, offset = worker.num_workers, worker.id
        wid = worker.id if worker is not None else 0
        seen = 0
        yielded = 0
        skip_reasons: Dict[str, int] = {}
        for i, row in enumerate(ds):
            if stride > 1 and (i % stride) != offset:
                continue
            if seen == 0:
                # Dump the real columns once so field-mapping bugs are obvious.
                log.info("[loader w%d] first row keys: %s", wid,
                         list(row.keys()))
            seen += 1
            try:
                sample, reason = self._process_row_diag(row)
            except Exception as exc:  # robust to occasional bad rows
                reason = f"exc:{type(exc).__name__}"
                sample = None
            if sample is not None:
                yielded += 1
                yield sample
            else:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

            # Progress heartbeat so the terminal shows life during the (slow)
            # streaming warmup, plus an early warning if nothing is usable.
            if seen % 50 == 0:
                log.info("[loader w%d] streamed=%d yielded=%d skips=%s",
                         wid, seen, yielded, skip_reasons)
            if seen == 200 and yielded == 0:
                log.warning("[loader w%d] 200 rows streamed, 0 usable samples! "
                            "Likely a field/audio mapping issue. skips=%s",
                            wid, skip_reasons)
