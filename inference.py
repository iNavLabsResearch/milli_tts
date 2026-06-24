#!/usr/bin/env python
"""milli_tts inference entrypoint.

    # one-shot
    python inference.py --voice S4259699400335456 \
        --text "সময়মতে ডেলিভাৰী দিয়াৰ বাবে বহুত ভাল লাগিল" --out out.wav

    # interactive (prompts for voice_id + text in a loop)
    python inference.py --interactive

    # list known voices
    python inference.py --list-voices

    # clone an unseen voice from a reference clip, then speak
    python inference.py --register-voice myvoice --reference ref.wav \
        --voice myvoice --text "Hello world" --out hello.wav
"""

from __future__ import annotations

import argparse

from milli_tts.bootstrap import bootstrap
from milli_tts.core.logger import get_logger
from milli_tts.inference import TTSInferenceEngine

log = get_logger("inference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="milli_tts inference")
    p.add_argument("--config", default="config.json")
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    p.add_argument("--voice", default=None, help="voice_id to speak with")
    p.add_argument("--text", default=None, help="text to synthesize")
    p.add_argument("--out", default="outputs/sample.wav", help="output wav path")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--list-voices", action="store_true")
    p.add_argument("--register-voice", default=None,
                   help="voice_id to create from --reference clip")
    p.add_argument("--reference", default=None, help="reference wav for cloning")
    p.add_argument("--interactive", action="store_true")
    return p.parse_args()


def _speak(engine: TTSInferenceEngine, text: str, voice: str, out: str,
           **kw) -> None:
    res = engine.synthesize(text, voice, **kw)
    res.save(out)
    log.info("Saved %s | frames=%d | first_chunk=%.1fms | total=%.1fms | "
             "RTF=%.2fx | voice=%s", out, res.num_frames, res.first_chunk_ms,
             res.latency_ms, res.real_time_factor, res.voice_id)


def main() -> None:
    args = parse_args()
    bootstrap(args.config)
    engine = TTSInferenceEngine(checkpoint=args.checkpoint)

    if args.list_voices:
        voices = engine.list_voices()
        print(f"\n{len(voices)} known voice_id(s):")
        for v in voices[:200]:
            print("  ", v)
        return

    if args.register_voice:
        if not args.reference:
            raise SystemExit("--register-voice requires --reference <wav>")
        engine.register_reference(args.register_voice, args.reference)

    kw = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    kw = {k: v for k, v in kw.items() if v is not None}

    if args.interactive:
        print("milli_tts interactive mode. Ctrl-C to exit.")
        default_voice = engine.cfg.voice.default_voice_id
        while True:
            try:
                voice = input(f"voice_id [{default_voice}]: ").strip() or default_voice
                text = input("text: ").strip()
                if not text:
                    continue
                out = f"outputs/interactive_{abs(hash(text)) % 10**8}.wav"
                _speak(engine, text, voice, out, **kw)
            except (KeyboardInterrupt, EOFError):
                print("\nbye")
                break
        return

    if args.text:
        _speak(engine, args.text, args.voice, args.out, **kw)
    else:
        log.info("Nothing to do. Pass --text, --interactive or --list-voices.")


if __name__ == "__main__":
    main()
