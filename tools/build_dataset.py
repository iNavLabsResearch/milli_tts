#!/usr/bin/env python
"""Build the Hindi IndicVoices speaker catalog + report the train/val split.

This makes the "perfect dataset" prep that real per-speaker TTS needs:

  1. **Speaker catalog** — every Hindi `speaker_id` is assigned a DENSE,
     collision-free embedding index (written to the VoiceBank's
     `voice_index.json`). Training + inference then agree on indices and each
     real voice gets its own embedding row (no hash collisions). Inference
     `--voice S42598...` resolves straight to that row.
  2. **Train/val split** — the split is deterministic (a per-utterance hash, see
     `huggingface.val_holdout_mod`); this tool just reports/sanity-checks the
     resulting sizes so you know what you're training on.

It reads only metadata (no audio decode). If `speaker_id` is a HF ClassLabel the
full speaker list comes straight from the schema (instant); otherwise it scans.

Usage (run once, before training; setup_colab.sh calls it for you):

    python tools/build_dataset.py                 # full pass
    python tools/build_dataset.py --max-rows 30000  # cap the scan (faster)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milli_tts.bootstrap import bootstrap  # noqa: E402
from milli_tts.core.logger import get_logger  # noqa: E402
from milli_tts.core.static_memory_cache import StaticMemoryCache  # noqa: E402
from milli_tts.data.dataset import IndicVoicesDataset  # noqa: E402

log = get_logger("build_dataset")


def main() -> bool:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--max-rows", type=int, default=0,
                   help="Cap rows scanned for stats/catalog (0 = unlimited).")
    args = p.parse_args()

    bootstrap(args.config)
    cfg = StaticMemoryCache.config()
    ds = IndicVoicesDataset(register_voices=False)
    bank = ds.voice_bank

    if ds.speaker_id_source != "row":
        log.warning("voice.speaker_id_source=%r (not 'row') — the catalog will "
                    "only hold the collapsed voices. Set it to 'row' for real "
                    "per-speaker voices.", ds.speaker_id_source)

    # Open the stream once so we learn the schema (does speaker_id carry the full
    # ClassLabel name list?) before deciding whether a full scan is needed.
    log.info("Opening Hindi stream to read the schema…")
    ds._hf_dataset = ds._build_hf_dataset()

    t0 = time.time()
    speakers: dict = {}            # speaker_id -> [gender, lang, n_utts]
    train_n = val_n = 0
    scanned = 0

    names = ds._speaker_names
    if names and ds.speaker_id_source == "row":
        log.info("speaker_id is a ClassLabel with %d names — registering the "
                 "whole catalog directly (sorted, no full scan needed).",
                 len(names))
        for sid in sorted({str(n).strip() for n in names if str(n).strip()}):
            bank.register_sequential(sid, lang=ds.default_lang)

    # Scan metadata for split sizes + per-speaker utterance counts (and to catch
    # any speakers not in the ClassLabel list). Capped by --max-rows for speed.
    log.info("Scanning metadata (no audio decode)%s…",
             f" up to {args.max_rows} rows" if args.max_rows else "")
    for spk, gender, lang, is_val in ds.iter_metadata(
            max_rows=args.max_rows or None):
        scanned += 1
        e = speakers.get(spk)
        if e is None:
            speakers[spk] = [gender, lang, 1]
        else:
            e[2] += 1
            if gender and not e[0]:
                e[0] = gender
        if is_val:
            val_n += 1
        else:
            train_n += 1
        if scanned % 5000 == 0:
            log.info("  scanned=%d speakers=%d train=%d val=%d (%.0fs)",
                     scanned, len(speakers), train_n, val_n, time.time() - t0)

    # Register any scan-only speakers (when speaker_id wasn't a ClassLabel).
    for spk in sorted(speakers):
        gender, lang, _ = speakers[spk]
        bank.register_sequential(spk, gender=gender, lang=lang or ds.default_lang)
    bank.save()

    total = train_n + val_n
    val_pct = (100.0 * val_n / total) if total else 0.0
    manifest = {
        "dataset_repo": cfg.huggingface.dataset_repo,
        "dataset_config": ds.dataset_config,
        "lang": ds.default_lang,
        "speaker_id_source": ds.speaker_id_source,
        "quality_filter": ds.quality_filter,
        "min_quality_decision": cfg.huggingface.min_quality_decision,
        "val_holdout_mod": ds.val_holdout_mod,
        "speakers_in_catalog": len(bank),
        "scanned_rows": scanned,
        "scanned_train": train_n,
        "scanned_val": val_n,
        "scanned_val_pct": round(val_pct, 2),
        "capped": bool(args.max_rows),
    }
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    mpath = os.path.join(cfg.paths.data_dir, "hindi_manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    log.info("Catalog: %d speakers -> %s", len(bank), bank.index_path)
    log.info("Split (scanned%s): train=%d  val=%d  (val≈%.1f%%)",
             " sample" if args.max_rows else "", train_n, val_n, val_pct)
    log.info("Manifest -> %s", mpath)
    if len(bank) == 0:
        log.error("No speakers catalogued — check dataset access / Hindi config.")
        return False
    return True


if __name__ == "__main__":
    ok = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)
