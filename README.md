# milli_tts

Production-grade, end-to-end **text-to-speech for Indian languages + English**,
built to **fine-tune on a Colab T4** and run **sub-100 ms streaming inference**.

It is an **RQ-Transformer** (Moshi/Mimi lineage) on top of the **frozen Mimi**
neural audio codec — the same recipe Kyutai uses for Pocket-TTS — adapted to the
[`ai4bharat/IndicVoices`](https://huggingface.co/datasets/ai4bharat/IndicVoices)
STT corpus, whose transcripts become TTS training labels.

> Full diagrams + rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (Mermaid).

```
text ids ─► text embedding ─┐
                            ├─► Backbone (temporal) Transformer ─► h_t
prev audio frame codes ─────┘        (causal: audio attends to all text)
+ speaker (voice_id) embedding                  │
                                                ▼
                                  Depth Transformer (over Q codebooks)
                                                │
                                                ▼
                              Q Mimi codes  ─► frozen Mimi decoder ─► 24 kHz wav
```

---

## Why this architecture (answers to the design questions)

* **No separate text encoder, no dimension mismatch.** Text and audio are both
  embedded internally to `d_model` and flow through *one* decoder-only backbone.
  Text is a causal prefix; audio frames attend back to it. That's what keeps
  latency low and removes the encoder/decoder dimension headaches.
* **The depth transformer** models the residual codebooks *within* a frame
  (the RQ-Transformer trick), so we don't need a MusicGen-style delay pattern.
* **Mimi is frozen** — we never train the codec, only predict its codes. This is
  what makes T4 training feasible and inference fast (12.5 Hz frame rate).
* **`voice_id` lives in the conditioning, not in Mimi.** A pretrained codec has
  no speaker concept, so we build the speaker catalog ourselves:
  * `EmbeddingTableConditioner` — every IndicVoices `speaker_id` (~400 of them)
    gets a learned embedding. Pass `voice_id="S4259699400335456"` at inference.
  * `PrefixConditioner` — feed a ~10 s reference clip to clone an **unseen**
    voice zero-shot.
* **Yes, STT data trains TTS.** Inverting the task (text → waveform) is exactly
  delayed-streams TTS. 400 speakers is a *feature*: each becomes a `voice_id`.

---

## Design patterns / structure

| Pattern | Where |
|---|---|
| **Singleton + lazy object store** | `core/StaticMemoryCache` — one config + shared codec/tokenizer/device |
| **Typed config (frozen dataclasses)** | `core/config.py`, secrets via `ENV:` markers |
| **Factory + Registry** | `models/factory.py` — build any arch by name |
| **Strategy** | `models/conditioning.py` — embedding-table vs prefix-clone voices |
| **Template Method** | `models/base.BaseTTSModel` contract |
| **Facade / Adapter** | `training/Trainer`, `training/WandbTracker` |

```
config.json                  # every path / repo / model / param / secret (ENV:)
train.py                     # `python train.py`  (Colab entrypoint)
inference.py                 # `python inference.py --interactive`
milli_tts/
  core/      static_memory_cache.py  config.py  logger.py  registry.py
  data/      mimi_codec.py  text_tokenizer.py  voice_bank.py  dataset.py  collator.py
  models/    layers.py  conditioning.py  rq_transformer.py  sampling.py  factory.py  base.py
  training/  trainer.py  optim.py  tracker.py  checkpoint.py
  inference/ engine.py
  utils/     audio.py  seed.py
tests/       smoke_test.py   # CPU, no-network end-to-end check
notebooks/   Colab_Train.ipynb
```

---

## Quickstart (Colab T4)

```python
!git clone https://github.com/iNavLabsResearch/milli_tts.git
%cd milli_tts
!bash setup_colab.sh

import os
os.environ["HF_TOKEN"]      = "hf_..."
os.environ["WANDB_API_KEY"] = "..."

!python train.py                       # live graphs in W&B
```

Inference:

```bash
python inference.py --interactive                      # prompts voice_id + text
python inference.py --voice S4259699400335456 \
                    --text "সময়মতে ডেলিভাৰী দিয়াৰ বাবে বহুত ভাল লাগিল" \
                    --out out.wav
python inference.py --list-voices
# zero-shot clone an unseen speaker:
python inference.py --register-voice myvoice --reference ref.wav \
                    --voice myvoice --text "Hello world" --out hello.wav
```

---

## Configuration

Everything is in **`config.json`**, loaded once through `StaticMemoryCache` and
read everywhere via `StaticMemoryCache.config()`. Secrets use `ENV:NAME` markers
so the JSON stays committable; set `HF_TOKEN` / `WANDB_API_KEY` in the env or a
`.env` file (see `.env.example`). Key knobs:

* `huggingface.dataset_config` — which IndicVoices language (e.g. `assamese`).
* `codec.num_codebooks` — 8 is a good quality/latency trade-off on T4.
* `training.batch_size` / `grad_accum_steps` / `precision` — `bf16` auto-falls
  back to `fp16` on a T4.
* `inference.target_latency_ms` — the 100 ms goal; `synthesize()` reports the
  measured `first_chunk_ms`, total latency, and real-time factor.

---

## Hitting sub-100 ms on a T4

The model is decoder-only with a tiny depth transformer, so per-frame cost is
small. Levers, in order of impact:

1. **fp16 + KV cache** (default) — text prefix is prefilled once, then each
   12.5 Hz frame is one backbone step + Q depth steps.
2. **Fewer codebooks** (`codec.num_codebooks = 8`) — fewer depth steps/frame.
3. **Smaller backbone** for the latency-critical tier (≈100 M, Pocket-TTS class).
4. **Stream in chunks** (`inference.stream_chunk_frames`) so time-to-first-audio
   is what the user perceives — that's the number kept under ~100 ms while the
   tail generates faster than real time (`real_time_factor > 1`).
5. Optional `inference.use_cuda_graphs` / `training.compile_model`.

---

## Notes

* Without `moshi` installed, `MimiCodec` falls back to a **dummy codec** so the
  pipeline (and `tests/smoke_test.py`) runs on a plain CPU box. Install `moshi`
  (in `requirements.txt`) for real 24 kHz audio.
* Generated audio quality will trail ElevenLabs — that's expected for a small
  model fine-tuned on STT data; the goal here is a working, fast, controllable
  Indic+English pipeline you fully own.

Run the offline check anytime:

```bash
python -m tests.smoke_test
```
