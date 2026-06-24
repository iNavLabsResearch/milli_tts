#!/usr/bin/env python
"""Quick streaming-data probe — verify IndicVoices yields usable samples.

Run this BEFORE a long training job to confirm the dataset streams and the
field mapping is correct (it bypasses the DataLoader, Mimi and the model):

    python tools/check_data.py            # pull 5 usable samples
    python tools/check_data.py --n 10

If it prints samples, training will get data. If it streams many rows but
yields 0, you'll see the skip reasons (no_text / no_audio / too_short / …) and
the real row keys, which tells you exactly what to fix in config/dataset.
"""

from __future__ import annotations

import argparse
import time

from milli_tts.bootstrap import bootstrap
from milli_tts.core.logger import get_logger
from milli_tts.data.dataset import IndicVoicesDataset

log = get_logger("check_data")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--n", type=int, default=5, help="usable samples to fetch")
    args = p.parse_args()

    bootstrap(args.config)
    ds = IndicVoicesDataset()  # single-process, no workers — clearest logs

    log.info("Streaming… (first sample can take a minute or two)")
    t0 = time.time()
    got = 0
    for sample in ds:
        got += 1
        log.info("sample %d | dur=%.2fs | spk=%s | lang=%s | text=%r",
                 got, sample["duration"], sample["speaker_id"],
                 sample["lang"], _short(sample))
        if got == 1:
            log.info("  -> first usable sample after %.1fs", time.time() - t0)
        if got >= args.n:
            break

    ok = got > 0
    if not ok:
        log.error("No usable samples produced — see skip reasons above.")
    else:
        log.info("OK: %d usable samples in %.1fs. Streaming works.",
                 got, time.time() - t0)
    return ok


def _short(sample) -> str:
    ids = sample["text_ids"].tolist()[:16]
    return f"<{len(sample['text_ids'])} tok> {ids}"


if __name__ == "__main__":
    import os
    import sys

    _ok = main()
    # Hard-exit to skip interpreter finalization — HF streaming leaves a C
    # prefetch thread alive that crashes during Py_Finalize (cosmetic
    # "PyGILState_Release … finalizing"). The work is already done here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if _ok else 1)
