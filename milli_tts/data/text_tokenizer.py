"""Text tokenizer wrapper for Indic + English (code-mixed) text.

We default to a **byte-level** tokenizer (ByT5) because it has zero
out-of-vocabulary risk across Assamese / Bengali / Tamil / Devanagari /
Latin scripts and handles code-mixing ("EV charger" inside a Hindi sentence)
for free — every character is just bytes. A SentencePiece/HF model can be
swapped in via ``config.json`` without touching call sites.

The wrapper exposes a stable interface (``encode`` / ``decode`` /
``vocab_size`` / special token ids) so the model never depends on which
underlying tokenizer is configured.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from milli_tts.core.logger import get_logger
from milli_tts.core.static_memory_cache import StaticMemoryCache

log = get_logger("data.tokenizer")


class TextTokenizer:
    def __init__(self, hf_tokenizer, max_len: int) -> None:
        self._tok = hf_tokenizer
        self.max_len = max_len
        # Resolve special ids with sensible fallbacks.
        self.pad_id = self._first_not_none(
            getattr(hf_tokenizer, "pad_token_id", None), 0)
        self.eos_id = self._first_not_none(
            getattr(hf_tokenizer, "eos_token_id", None), 1)
        self.bos_id = self._first_not_none(
            getattr(hf_tokenizer, "bos_token_id", None), self.eos_id)
        self.unk_id = self._first_not_none(
            getattr(hf_tokenizer, "unk_token_id", None), self.pad_id)

    @staticmethod
    def _first_not_none(*vals):
        for v in vals:
            if v is not None:
                return int(v)
        return 0

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls) -> "TextTokenizer":
        def _factory() -> "TextTokenizer":
            cfg = StaticMemoryCache.config()
            tcfg = cfg.tokenizer
            tok = cls._load_hf(tcfg.hf_repo, cfg.huggingface.token)
            if tok is None and tcfg.fallback_repo:
                log.warning("Primary tokenizer failed; trying fallback %s",
                            tcfg.fallback_repo)
                tok = cls._load_hf(tcfg.fallback_repo, cfg.huggingface.token)
            if tok is None:
                log.warning("All HF tokenizers failed; using raw byte tokenizer.")
                tok = _ByteTokenizer()
            log.info("Text tokenizer ready (vocab=%d)", _safe_vocab(tok))
            return cls(tok, tcfg.max_text_len)

        return StaticMemoryCache.get_or_create("tokenizer::text", _factory)

    @staticmethod
    def _load_hf(repo: str, token: Optional[str]):
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(repo, token=token,
                                                 trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - env dependent
            log.warning("Failed to load HF tokenizer %s: %s", repo, exc)
            return None

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return _safe_vocab(self._tok)

    def encode(self, text: str, *, add_eos: bool = True,
               truncate: bool = True) -> List[int]:
        ids = self._tok.encode(text) if hasattr(self._tok, "encode") else self._tok(text)
        if truncate and len(ids) > self.max_len - 1:
            ids = ids[: self.max_len - 1]
        if add_eos and (not ids or ids[-1] != self.eos_id):
            ids = ids + [self.eos_id]
        return ids

    def encode_tensor(self, text: str, **kw) -> torch.Tensor:
        return torch.tensor(self.encode(text, **kw), dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        ids = [int(i) for i in ids if int(i) != self.pad_id]
        try:
            return self._tok.decode(ids, skip_special_tokens=True)
        except Exception:
            return self._tok.decode(ids)


def _safe_vocab(tok) -> int:
    for attr in ("vocab_size",):
        v = getattr(tok, attr, None)
        if isinstance(v, int):
            return v
    try:
        return len(tok)
    except Exception:
        return 256


class _ByteTokenizer:
    """Absolute last-resort tokenizer: UTF-8 bytes + 3 special ids."""

    pad_token_id = 256
    eos_token_id = 257
    bos_token_id = 258
    unk_token_id = 256
    vocab_size = 259

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        bs = bytes([i for i in ids if i < 256])
        return bs.decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return self.vocab_size
