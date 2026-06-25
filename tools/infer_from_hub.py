#!/usr/bin/env python
"""Pull the latest checkpoint from HF and run inference for all voice × text pairs.

Usage (on the GPU server):
    python tools/infer_from_hub.py

Outputs go to outputs/hub_samples/. Download them with:
    docker cp <container>:/milli_tts/outputs/hub_samples/ ./hub_samples/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milli_tts.bootstrap import bootstrap
from milli_tts.core.logger import get_logger
from milli_tts.core.static_memory_cache import StaticMemoryCache
from milli_tts.inference import TTSInferenceEngine

log = get_logger("infer_from_hub")

# ── texts ────────────────────────────────────────────────────────────────
TEXTS = [
    ("en", "Today is a beautiful day, and the world is full of possibilities."),
    ("hi", "आज का दिन बहुत सुंदर है, और दुनिया संभावनाओं से भरी हुई है।"),
]

# ── voices (IndicTTS-Hindi derives these from gender) ────────────────────
VOICES = ["hi_female", "hi_male"]

OUT_DIR = os.path.join(str(_REPO_ROOT), "outputs", "hub_samples")


def pull_latest_checkpoint() -> str:
    """Download the most recent latest.pt from the HF checkpoint repo."""
    from huggingface_hub import HfApi, hf_hub_download

    cfg = StaticMemoryCache.config()
    raw_repo = cfg.huggingface.checkpoint_repo
    token = cfg.huggingface.token

    api = HfApi(token=token)

    # checkpoint_repo in config may be a bare name ("milli_tts_weights") —
    # resolve it to the full "owner/repo" via the authenticated user, matching
    # what hf_sync.py's create_repo() does during training.
    if "/" not in raw_repo:
        user = api.whoami(token=token)["name"]
        repo = f"{user}/{raw_repo}"
        log.info("Resolved checkpoint repo -> %s", repo)
    else:
        repo = raw_repo

    files = api.list_repo_files(repo, repo_type="model")
    latest_files = sorted(
        [f for f in files if f.endswith("/latest.pt")], reverse=True)
    if not latest_files:
        raise FileNotFoundError(
            f"No latest.pt found in HF repo '{repo}'. Files: {files[:20]}")
    remote_path = latest_files[0]
    log.info("Pulling %s from hf://%s …", remote_path, repo)
    local = hf_hub_download(repo, remote_path, repo_type="model", token=token)
    log.info("Downloaded -> %s", local)
    return local


def ensure_voice_bank():
    """Populate the voice bank with the two IndicTTS-Hindi voices so
    inference can resolve them even without a trained voice_index.json."""
    from milli_tts.data.voice_bank import VoiceBank
    bank = VoiceBank.from_config()
    for vid in VOICES:
        if vid not in bank:
            gender = vid.split("_")[-1] if "_" in vid else None
            bank.add_or_get(vid, gender=gender, lang="hi")
            log.info("Registered voice '%s' in bank (index=%d).",
                     vid, bank.index_of(vid))
    bank.save()


def main():
    bootstrap("config.json")
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt_path = pull_latest_checkpoint()
    ensure_voice_bank()
    engine = TTSInferenceEngine(checkpoint=ckpt_path)

    log.info("Generating %d texts × %d voices = %d samples …",
             len(TEXTS), len(VOICES), len(TEXTS) * len(VOICES))
    saved = []
    for voice_id in VOICES:
        for lang, text in TEXTS:
            tag = f"{voice_id}_{lang}"
            out_path = os.path.join(OUT_DIR, f"{tag}.wav")
            log.info("──── %s: \"%s\"", tag, text[:60])
            res = engine.synthesize(text, voice_id)
            res.save(out_path)
            log.info("  -> %s | frames=%d | latency=%.0fms | first_chunk=%.0fms "
                     "| RTF=%.2fx", out_path, res.num_frames, res.latency_ms,
                     res.first_chunk_ms, res.real_time_factor)
            saved.append(out_path)

    print(f"\n{'='*60}")
    print(f"Done. {len(saved)} files in {OUT_DIR}/:")
    for p in saved:
        print(f"  {os.path.basename(p)}")
    print(f"\nTo download from a Docker container:")
    print(f"  docker cp <container>:{OUT_DIR}/ ./hub_samples/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
