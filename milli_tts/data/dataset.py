"""Hugging Face TTS dataset adapter (ai4bharat/IndicVoices, Hindi).

Primary target: **ai4bharat/IndicVoices** (config ``hindi``) — a large *field*
STT corpus (audio + ``normalized``/``verbatim`` transcript + ``speaker_id`` +
``gender`` + ``lang`` + a per-row ``verification_report`` of QC flags). For TTS
we invert the task — **text is the input, the spoken waveform is the target**.

Because IndicVoices is spontaneous speech (not studio TTS), two filters matter:
* ``quality_filter`` drops noisy / mispronounced / text-mismatched clips using
  the ``verification_report`` (see :data:`_BAD_QC_FLAGS`).
* ``speaker_id_source="gender"`` collapses the many field speakers to a few
  stable voices (``hi_female`` / ``hi_male``) so the conditioning trains well.

It stays backward-compatible with single-language TTS corpora such as
**SPRINGLab/IndicTTS-Hindi** (``audio`` + ``text`` + ``gender``, no per-row
``lang``), which fall back to ``default_lang`` and the gender-derived speaker.

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

import ast
import os
import re
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
# `normalized` is the cleaned transcript (no [uhh]/[horn] disfluency tags);
# prefer it over `verbatim`/`unsanitized_*`.
_TEXT_FIELDS = ("normalized", "text", "verbatim", "sentence")
_SPEAKER_FIELDS = ("speaker_id", "speaker", "client_id")

# Inline annotation tags that survive in some transcripts, in TWO bracket
# styles: square "[uhh]"/"[horn]" and angle "<breathing>"/"<inhaling>"/
# "<Persistent-noise-start>". Stripped so the model never "speaks" a tag.
_TAG_RE = re.compile(r"\[[^\]]*\]|<[^>]*>")

# IndicVoices verification_report flags that make a clip unfit for TTS — either
# the AUDIO is bad (noise / echo / unclear / low volume / chatter) or the TEXT
# does not match the audio (skipping / repeating / wrong prompt / reading it),
# or the SPEAKER label is wrong (wrong_gender / duplicate_speaker — these poison
# per-speaker conditioning). Training on these is what produces garbled speech.
_BAD_QC_FLAGS = (
    "unclear_audio", "noise_persistent", "noise_intermittent", "echo_present",
    "low_volume", "chatter_persistent", "chatter_intermittent",
    "mispronunciation", "wrong_language", "stretching", "skipping_words",
    "repeating_content", "incorrect_text_prompt", "reading_prompt",
    "bad_extempore_quality", "wrong_gender", "duplicate_speaker",
)
# Overall human decision, best -> worst. Unknown values rank high (= keep).
_DECISION_RANK = {"excellent": 3, "good": 2, "average": 1, "poor": 0}


def _clean_text(text: str) -> str:
    """Drop bracketed/angle disfluency+noise tags and collapse whitespace."""
    return " ".join(_TAG_RE.sub(" ", text).split())

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
                 register_voices: bool = True,
                 role: str = "train", val_size: int = 0) -> None:
        super().__init__()
        cfg = StaticMemoryCache.config()
        self.cfg = cfg
        self.hf = cfg.huggingface
        self.split = split or self.hf.dataset_split
        self.target_sr = cfg.codec.sample_rate
        self.min_sec = cfg.training.min_audio_seconds
        self.max_sec = cfg.training.max_audio_seconds
        self.register_voices = register_voices
        # role="val" yields the first `val_size` stream rows as a held-out set;
        # role="train" skips those same rows so the two never overlap.
        self.role = role
        self.val_size = int(val_size)
        # Train-stream shuffle: seed (for reproducibility) + per-epoch counter so
        # each pass over a small corpus is reshuffled instead of replayed in order.
        self.seed = int(getattr(cfg.project, "seed", 1337))
        self._epoch = 0
        self.tokenizer = tokenizer or TextTokenizer.from_config()
        self.voice_bank = voice_bank or VoiceBank.from_config()
        self._hf_dataset = None
        self._gender_names = None  # filled from the gender ClassLabel feature
        self._speaker_names = None  # filled if speaker_id is a ClassLabel

        # ---- quality + speaker policy -------------------------------- #
        self.is_indicvoices = "indicvoices" in str(self.hf.dataset_repo).lower()
        self.quality_filter = bool(getattr(self.hf, "quality_filter", False))
        self.min_decision_rank = _DECISION_RANK.get(
            str(getattr(self.hf, "min_quality_decision", "good")).lower(), 2)
        # "row" -> one learned embedding per raw speaker_id (real per-speaker
        # voices, what inference selects); "gender" -> collapse to hi_female/
        # hi_male.
        self.speaker_id_source = str(
            getattr(cfg.voice, "speaker_id_source", "row")).lower()
        # Deterministic utterance-level train/val split: a row is VAL iff
        # hash(speaker_id|text) % val_holdout_mod == 0 (≈1/mod held out). Keyed on
        # the UTTERANCE (not the speaker) so val speakers are still seen in train
        # — essential for the embedding-table conditioning to have a meaningful
        # val loss. 0 disables (falls back to the take/skip prefix split).
        self.val_holdout_mod = int(getattr(self.hf, "val_holdout_mod", 0) or 0)

        # ---- language selection -------------------------------------- #
        # `languages` from config decides which language(s) to keep.
        self.allowed_langs = _normalize_langs(getattr(self.hf, "languages", None))
        self.dataset_config = self.hf.dataset_config

        # IndicVoices ships one HF *config per language*, so a single requested
        # language also auto-selects the matching `dataset_config`. This remap is
        # IndicVoices-specific — other corpora (e.g. SPRINGLab/IndicTTS-Hindi,
        # config "default") must keep their own config name, so gate it on repo.
        is_indicvoices = self.is_indicvoices
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
            # Layout-robust: some IndicVoices snapshots expose one HF *config per
            # language* ("hindi"), others ship a single config with a per-row
            # `lang` column. If the requested config name isn't valid, retry
            # WITHOUT a config and lean on the per-row language filter instead —
            # `allowed_langs` (= ["hi"]) still keeps it Hindi-only.
            config_err = any(k in msg for k in (
                "config name", "builderconfig", "unknown config",
                "not a valid config", "available configs", "config_name",
                "load_dataset_builder"))
            if config_err and self.dataset_config:
                log.warning("Config '%s' not valid for %s (%s) — retrying with "
                            "no config and filtering by per-row lang=%s.",
                            self.dataset_config, self.hf.dataset_repo, exc,
                            sorted(self.allowed_langs) or "ALL")
                self.dataset_config = None
                ds = load_dataset(self.hf.dataset_repo, **kwargs)
            elif any(k in msg for k in ("gated", "403", "401", "authenticated",
                                        "access", "permission")):
                raise RuntimeError(
                    f"Cannot access {self.hf.dataset_repo}: it is GATED. Open "
                    f"https://huggingface.co/datasets/{self.hf.dataset_repo} "
                    f"while logged in as the token owner and click 'Agree and "
                    f"access', then retry. Original error: {exc}") from exc
            else:
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
        # `speaker_id` may also be a ClassLabel (streams as an int). Capture its
        # names so per-speaker conditioning maps the int back to the real id
        # string (e.g. "S4259869900354210") instead of silently dropping it.
        sfeat = feats.get("speaker_id") if feats else None
        self._speaker_names = list(getattr(sfeat, "names", None) or []) or None

        # Held-out validation split.
        #  * val_holdout_mod>0 (default for IndicVoices): a deterministic
        #    UTTERANCE-hash split applied per-row in _process_row_diag — disjoint,
        #    representative across all speakers, and val speakers stay in train.
        #    No take/skip here (the whole stream feeds both roles, filtered).
        #  * else: legacy prefix split — first `val_size` rows are val, train
        #    skips them.
        if self.val_holdout_mod <= 0 and self.val_size > 0:
            if self.role == "val":
                ds = ds.take(self.val_size)
            elif self.role == "train":
                ds = ds.skip(self.val_size)

        # Break long contiguous speaker/length blocks in the train stream. Without
        # this the model over-specializes to whatever block it's currently on,
        # which surfaces as a growing per-epoch sawtooth in the (fixed) val
        # metrics and wastes capacity on memorization. Reservoir shuffle over a
        # buffer; reshuffled per epoch via set_epoch() in __iter__. Train only —
        # val stays a deterministic, disjoint held-out prefix.
        buf = int(getattr(self.hf, "shuffle_buffer_size", 0) or 0)
        if (self.role == "train" and self.hf.streaming and buf > 0
                and hasattr(ds, "shuffle")):
            ds = ds.shuffle(seed=self.seed, buffer_size=buf)
            log.info("Train stream shuffled (buffer_size=%d, seed=%d) — avoids "
                     "block-ordered batches that cause the val sawtooth.",
                     buf, self.seed)

        log.info("Stream ready (role=%s, audio col=%s, gender_names=%s, "
                 "self-decode). Pulling rows…", self.role, audio_col,
                 self._gender_names)
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

    def _quality_ok(self, row: Dict) -> bool:
        """IndicVoices QC gate: drop noisy / mismatched clips.

        Reads the per-row ``verification_report`` (a dict, or a stringified
        dict when streamed as text). Rows below ``min_quality_decision`` or with
        any audio/text-mismatch flag set are rejected. No report -> keep (so
        corpora without this column are unaffected).
        """
        if not (self.quality_filter and self.is_indicvoices):
            return True
        report = row.get("verification_report")
        if report is None:
            return True
        if isinstance(report, str):
            try:
                report = ast.literal_eval(report)
            except Exception:
                return True
        if not isinstance(report, dict):
            return True
        decision = str(report.get("decision", "")).strip().lower()
        if decision and _DECISION_RANK.get(decision, 99) < self.min_decision_rank:
            return False
        return not any(report.get(flag) is True for flag in _BAD_QC_FLAGS)

    def _is_val_row(self, speaker_id: str, text: str) -> bool:
        """Deterministic per-utterance val membership: ``hash(spk|text)%mod==0``."""
        import hashlib
        key = f"{speaker_id}|{text}".encode("utf-8")
        h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        return (h % self.val_holdout_mod) == 0

    def _speaker_name(self, value) -> Optional[str]:
        """Normalize a speaker_id cell to its string id (decode ClassLabel ints)."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, str):
            return value.strip() or None
        try:
            idx = int(value)
        except (TypeError, ValueError):
            return None
        if self._speaker_names and 0 <= idx < len(self._speaker_names):
            return str(self._speaker_names[idx]).strip()
        return str(idx)

    def _row_speaker_id(self, row: Dict, lang_code: str, gender) -> str:
        """Resolve the speaker_id used for conditioning, per `speaker_id_source`."""
        if self.speaker_id_source == "gender":
            return f"{lang_code or 'hi'}_{gender or 'unknown'}"
        raw = row.get("speaker_id")
        if raw is None:
            raw = row.get("speaker") if row.get("speaker") is not None \
                else row.get("client_id")
        sid = self._speaker_name(raw)
        return sid or f"{lang_code or 'spk'}_{gender or 'unknown'}"

    def _process_row_diag(self, row: Dict):
        """Like _process_row but also returns a short skip reason for diagnostics."""
        text = self._pick(row, _TEXT_FIELDS)
        if not text:
            return None, "no_text"
        # Strip residual [uhh]/<breathing>/<noise> tags from the transcript.
        text = _clean_text(text)
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

        # Quality gate (cheap, before the expensive audio decode).
        if not self._quality_ok(row):
            return None, "low_quality"

        gender = self._gender_name(row.get("gender"))
        # Speaker identity (real per-speaker id when speaker_id_source="row").
        speaker_id = self._row_speaker_id(row, lang_code, gender)

        # Train/val routing (cheap, BEFORE decode): keep only this role's rows.
        if self.val_holdout_mod > 0:
            is_val = self._is_val_row(speaker_id, text)
            if self.role == "train" and is_val:
                return None, "held_out_val"
            if self.role == "val" and not is_val:
                return None, "not_val"

        # Cheap duration pre-filter on the `duration` column (skip decode if it's
        # already out of range). The exact length is recomputed from the wav.
        dfield = row.get("duration")
        if isinstance(dfield, (int, float)) and dfield > 0:
            if dfield < self.min_sec:
                return None, "too_short"
            if dfield > self.max_sec:
                return None, "too_long"

        wav = self._extract_wav(row)
        if wav is None or wav.numel() == 0:
            return None, "no_audio"
        dur = wav.numel() / self.target_sr
        if dur < self.min_sec:
            return None, "too_short"
        if dur > self.max_sec:
            return None, "too_long"

        if self.register_voices:
            spk_index = self.voice_bank.add_or_get(speaker_id, gender=gender,
                                                   lang=lang_code)
        else:
            # Deterministic fallback (same hashing as add_or_get) so an
            # not-yet-registered voice still maps to its real embedding row.
            spk_index = (self.voice_bank.index_of(speaker_id)
                         if speaker_id in self.voice_bank
                         else self.voice_bank.stable_index(speaker_id))

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
    def iter_metadata(self, max_rows: Optional[int] = None):
        """Yield ``(speaker_id, gender, lang_code, is_val)`` for quality-passing
        rows WITHOUT decoding audio.

        Used by ``tools/build_dataset.py`` to pre-build the speaker catalog and
        count the train/val split cheaply. Drops the audio column up front so we
        never pay the soundfile decode while cataloguing.
        """
        if self._hf_dataset is None:
            self._hf_dataset = self._build_hf_dataset()
        ds = self._hf_dataset
        audio_col = self._find_audio_col(ds)
        try:
            if audio_col and hasattr(ds, "remove_columns"):
                ds = ds.remove_columns([audio_col])
        except Exception:
            pass
        seen = 0
        for row in ds:
            seen += 1
            if max_rows and seen > max_rows:
                break
            text = self._pick(row, _TEXT_FIELDS)
            if not text:
                continue
            text = _clean_text(text)
            if not text:
                continue
            lang = row.get("lang") or self.default_lang
            lang_code = str(lang).strip().lower()
            lang_code = _CONFIG_TO_LANG_CODE.get(lang_code, lang_code)
            if self.allowed_langs and lang_code not in self.allowed_langs:
                continue
            if not self._quality_ok(row):
                continue
            gender = self._gender_name(row.get("gender"))
            speaker_id = self._row_speaker_id(row, lang_code, gender)
            is_val = (self.val_holdout_mod > 0
                      and self._is_val_row(speaker_id, text))
            yield speaker_id, gender, lang_code, is_val

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[Dict]:
        if self._hf_dataset is None:
            self._hf_dataset = self._build_hf_dataset()
        ds = self._hf_dataset
        # Reshuffle the streaming buffer each epoch (train only) so repeated
        # passes over a small corpus don't replay an identical order and memorize
        # it. No-op if the stream wasn't shuffled (buffer disabled / val role).
        if (self.role == "train" and self.hf.streaming
                and hasattr(ds, "set_epoch")):
            ds.set_epoch(self._epoch)
            self._epoch += 1
        # Shard the stream so each (DDP rank × DataLoader worker) sees a DISJOINT
        # slice — otherwise every GPU/worker would replay the same rows. Total
        # shards = world_size × num_workers; this worker's global shard index is
        # rank·num_workers + worker_id. Rank/size come from env (set by DDP) so
        # they're correct inside forked workers too. Validation (role="val") is
        # run on a single process over the full take()-set, so it isn't sharded.
        worker = torch.utils.data.get_worker_info()
        nw = worker.num_workers if worker is not None else 1
        wid = worker.id if worker is not None else 0
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        stride, offset = 1, 0
        total_shards = world * nw if self.role == "train" else 1
        shard_index = rank * nw + wid
        if self.hf.streaming and total_shards > 1:
            if hasattr(ds, "shard"):
                ds = ds.shard(num_shards=total_shards, index=shard_index)
            else:  # fallback: stride the iterator (no duplication across shards)
                stride, offset = total_shards, shard_index
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
