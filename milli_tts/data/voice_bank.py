"""VoiceBank — the catalog that turns a ``voice_id`` string into conditioning.

This is the answer to "ElevenLabs/Sarvam take a voice_id, but a pretrained
codec has no speaker concept". The speaker identity lives **here**, in the TTS
conditioning, not in Mimi.

Two complementary mechanisms (Strategy pattern, see ``models/conditioning.py``):

1. **Embedding table** — every distinct ``speaker_id`` in the training corpus
   (e.g. IndicVoices' ~400 speakers ``S4259699400335456`` …) is assigned a
   stable integer index. The model learns one embedding per index. At inference
   you pass ``voice_id="S42596..."`` and we look up the index.

2. **Reference prefix (zero-shot cloning)** — a ~10s reference clip is encoded
   to Mimi codes once and stored; at inference the model is primed with that
   prefix to clone an unseen voice.

The VoiceBank owns the ``speaker_id -> index`` mapping and persists it as JSON
so training and inference agree on indices. It is append-only: new speakers get
new indices, existing ones keep theirs.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch

from milli_tts.core.logger import get_logger
from milli_tts.core.static_memory_cache import StaticMemoryCache

log = get_logger("data.voice_bank")


@dataclass
class VoiceEntry:
    speaker_id: str
    index: int
    gender: Optional[str] = None
    lang: Optional[str] = None
    num_utterances: int = 0
    has_prefix: bool = False
    meta: Dict[str, str] = field(default_factory=dict)


class VoiceBank:
    def __init__(self, max_speakers: int, store_dir: str) -> None:
        self.max_speakers = max_speakers
        self.store_dir = store_dir
        self._entries: Dict[str, VoiceEntry] = {}
        self._lock = threading.RLock()
        os.makedirs(store_dir, exist_ok=True)

    # A threading.RLock can't be pickled, which breaks sending a dataset that
    # references this bank to *spawned* DataLoader workers. Drop the lock on
    # pickle and recreate it on the other side.
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls) -> "VoiceBank":
        def _factory() -> "VoiceBank":
            cfg = StaticMemoryCache.config()
            bank = cls(cfg.voice.max_speakers, cfg.paths.voice_bank_dir)
            bank.load()
            return bank

        return StaticMemoryCache.get_or_create("voice_bank", _factory)

    # ------------------------------------------------------------------ #
    @property
    def index_path(self) -> str:
        return os.path.join(self.store_dir, "voice_index.json")

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, speaker_id: str) -> bool:
        return speaker_id in self._entries

    # ------------------------------------------------------------------ #
    def stable_index(self, speaker_id: str) -> int:
        """Deterministic ``speaker_id -> index`` via a content hash.

        Crucially this is **independent of insertion order**, so every
        DistributedDataParallel rank maps a given speaker to the *same*
        embedding row even though each rank streams a different shard and meets
        speakers in a different order. (A sequential ``len(entries)`` index would
        desync the speaker embedding across ranks and corrupt its gradients.)

        Collisions are possible in principle but vanishingly unlikely while the
        speaker count stays well under ``max_speakers`` (e.g. 2 here); two
        colliding ids would simply share one voice embedding.
        """
        import hashlib

        h = hashlib.blake2b(speaker_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.max_speakers

    def add_or_get(self, speaker_id: str, *, gender: Optional[str] = None,
                   lang: Optional[str] = None) -> int:
        """Return the stable integer index for ``speaker_id`` (assigning if new)."""
        with self._lock:
            entry = self._entries.get(speaker_id)
            if entry is None:
                if len(self._entries) >= self.max_speakers:
                    log.warning(
                        "VoiceBank has %d speakers (>= max_speakers=%d); new "
                        "voices will collide on shared embeddings. Raise "
                        "voice.max_speakers in config.json.",
                        len(self._entries), self.max_speakers)
                entry = VoiceEntry(speaker_id=speaker_id,
                                   index=self.stable_index(speaker_id),
                                   gender=gender, lang=lang)
                self._entries[speaker_id] = entry
            entry.num_utterances += 1
            if gender and not entry.gender:
                entry.gender = gender
            if lang and not entry.lang:
                entry.lang = lang
            return entry.index

    def index_of(self, speaker_id: str) -> int:
        if speaker_id not in self._entries:
            raise KeyError(
                f"Unknown voice_id '{speaker_id}'. Known voices: "
                f"{self.list_voice_ids()[:10]} ... ({len(self)} total)")
        return self._entries[speaker_id].index

    def get(self, speaker_id: str) -> Optional[VoiceEntry]:
        return self._entries.get(speaker_id)

    def list_voice_ids(self) -> List[str]:
        return sorted(self._entries, key=lambda s: self._entries[s].index)

    # ------------------------------------------------------------------ #
    # Reference-prefix storage (zero-shot cloning)
    # ------------------------------------------------------------------ #
    def prefix_path(self, speaker_id: str) -> str:
        safe = speaker_id.replace("/", "_")
        return os.path.join(self.store_dir, f"prefix_{safe}.pt")

    def save_prefix(self, speaker_id: str, mimi_codes: torch.Tensor) -> None:
        """Persist a reference clip's Mimi codes ``[Q, frames]`` for cloning."""
        with self._lock:
            torch.save(mimi_codes.cpu(), self.prefix_path(speaker_id))
            entry = self._entries.get(speaker_id)
            if entry is None:
                self.add_or_get(speaker_id)
                entry = self._entries[speaker_id]
            entry.has_prefix = True
            self.save()

    def load_prefix(self, speaker_id: str) -> Optional[torch.Tensor]:
        path = self.prefix_path(speaker_id)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
        return None

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        with self._lock:
            payload = {
                "max_speakers": self.max_speakers,
                "entries": [asdict(e) for e in self._entries.values()],
            }
            tmp = self.index_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.index_path)
            log.info("Saved voice bank (%d voices) -> %s", len(self),
                     self.index_path)

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            return
        with open(self.index_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        with self._lock:
            self._entries = {}
            for d in payload.get("entries", []):
                e = VoiceEntry(**d)
                self._entries[e.speaker_id] = e
        log.info("Loaded voice bank (%d voices) from %s", len(self),
                 self.index_path)
