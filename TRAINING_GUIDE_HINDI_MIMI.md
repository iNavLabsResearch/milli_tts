# Training Guide: Hindi TTS with Mimi (IndicVoices-R)

This guide is kept in sync with the code. Where an earlier draft made claims that
the codebase already handled (or handled differently), those are corrected below.

## What the Architecture Does (Quick Summary)

```
Hindi text  →  TextEmbedder  ─┐
                               ├──► Backbone Transformer (8L, d=768, causal)
prev audio frame  ─────────────┘              │
+ speaker vector                               ▼
                                        h_audio [B, Ta, D]
                                               │
                                               ▼
                                    DepthHead (4L, causal over Q=8)
                                    predicts cb0 → cb1 → … → cb7
                                               │
                                               ▼
                                    Mimi.decode()  →  24 kHz wav
```

- **Backbone** = causal Transformer over the joint [text | audio] sequence.
  Text acts as a prefix; every audio frame attends to all text, no cross-attention.
- **DepthHead** = tiny causal Transformer over the 8 RVQ codebooks of one frame.
  This removes the need for MusicGen-style codebook delay patterns.
- **Mimi** is always **frozen**. The model only predicts integer codes; Mimi decodes them.

Two model implementations are registered and interchangeable (`model.arch`):
`rq_transformer_tts` (v1) and `rq_transformer_tts_v2` (modular rewrite — same
`compute_loss`/`generate`/checkpoint format). Training uses v1 by default.

---

## Dataset: SPRINGLab/IndicVoices-R_Hindi

This project trains on **`SPRINGLab/IndicVoices-R_Hindi`** — the cleaned / restored
("‑R") TTS variant of IndicVoices, Hindi only. It is a **single-config** repo, so:

- `huggingface.dataset_config` must be **`null`** (not `"hindi"`). The per-language
  config remap only applies to the multi-config `ai4bharat/IndicVoices`.
- **Text input = the `normalized` column** (cleaned transcript, no `[uhh]`/`<noise>`
  tags). Already the first candidate in `data/dataset.py::_TEXT_FIELDS`.
- **Audio = the `audio` column**, decoded with soundfile/librosa (no torchcodec).
- There is **no `verification_report` column**, so `quality_filter` /
  `min_quality_decision` are a no-op here. The "get clean audio" lever for ‑R is
  `huggingface.min_snr` / `huggingface.max_cer` (off by default — ‑R is already
  clean; set e.g. `min_snr: 15.0`, `max_cer: 0.5` to drop the worst clips).

### Access + token (this was the `DatasetNotFoundError` you hit)

The HF token lives in `config.json` (`huggingface.token`, REV-encoded) and
`bootstrap()` exports it to `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`. Every real
entrypoint (`train.py`, `inference.py`, `tools/build_dataset.py`) calls
`bootstrap()`, so they authenticate automatically. `setup_colab.sh` now also
bootstraps before its access check, so it reads the **configured** repo + token
instead of a hardcoded `ai4bharat/IndicVoices` with an empty `HF_TOKEN`.

If a check still fails: confirm `huggingface.token` is a valid `hf_…` token and,
if the chosen repo is gated, that you accepted its terms on the Hub while logged
in as that token's owner.

---

## macOS / DataLoader / MPS notes

- **The `cannot pickle '_thread.RLock'` crash is already guarded against.** The
  trainer forces the `fork` start method for workers (`_worker_mp_context`), and
  `VoiceBank` drops its `RLock` on pickle (`__getstate__`/`__setstate__`). You do
  **not** need to rewrite `_build_loader`. If you still hit a pickling error on a
  Mac smoke test, just set `"num_workers": 0` in `config.json` — don't remove the
  fork-forcing logic.
- **MPS precision**: `_resolve_precision` returns `float32` on non-CUDA devices
  (MPS has no usable bf16/fp16 AMP) — correct, but ~3–5× slower than a T4. Use
  Mac only for short smoke tests; train on Colab/Kaggle (T4) or better.

---

## Why acc_cb0 Stays Low (this part of the original guide is correct)

| Codebook | What it captures | Difficulty | Approx acc @ 5k steps |
|----------|-----------------|------------|------------------------|
| cb0 | Semantic content (WavLM-distilled) | ★★★★★ hardest | 5–15% |
| cb1 | Coarse acoustic residual | ★★★★ | 20–40% |
| cb2–cb5 | Mid acoustic residuals | ★★★ | 40–65% |
| cb6–cb7 | Fine texture | ★★ easiest | 60–80% |

Mimi distills WavLM/HuBERT features into **cb0**, making it a *semantic* token
(what is being said + prosody). Predicting the exact cb0 index from text is
high-entropy, many valid realizations. Random chance = 1/2048 ≈ **0.05%**; even a
well-trained model lands ~20–35%. Low early acc_cb0 is **normal**.

### Levers that actually help cb0 (corrected)

1. **Weight the depth transformer toward cb0** — `model.codebook_loss_weights`
   (see next section). This is the cheapest, no-slowdown lever and the
   recommended first move.
2. **Don't collapse speakers to gender.** The earlier guide suggested
   `speaker_id_source="gender"` — that **conflicts** with wanting real per-speaker
   voices, and isn't needed for IndicVoices‑R Hindi (a few hundred speakers, each
   with many utterances, train fine). Keep `speaker_id_source="row"` and make
   indices collision-free by pre-building the catalog (`tools/build_dataset.py`,
   which `setup_colab.sh` runs).
3. **label_smoothing** (e.g. 0.1) can curb over-confidence, but it inflates the
   reported CE; keep it 0 if you want `train/val ppl` to be a true perplexity.
4. **Train longer.** 6k steps is the floor; 20k+ for naturalness.

---

## Lever: weight the depth transformer (`codebook_loss_weights`)

The depth transformer predicts **all** Q codebooks under one cross-entropy. The
loss is now a per-codebook **weighted** CE so depth capacity focuses where it
matters — **without adding parameters, so training speed is unchanged**.

```json
"model": {
    "arch": "rq_transformer_tts",
    "codebook_loss_weights": [2.0, 1.4, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
}
```

- Up-weights **cb0** (semantic / intelligibility) and the coarse codebooks;
  down-weights fine texture (cb6/cb7), which the model learns easily and which
  matter least for intelligibility.
- Weights are **renormalized to mean 1** internally, so loss scale and perplexity
  stay comparable to a uniform run (init loss still ≈ ln(2048) ≈ 7.62).
- `null` / omitted = uniform (original behavior). A list of the wrong length is a
  hard error. Implemented in both v1 and v2
  (`RQTransformerTTS._build_codebook_weights` + weighted CE in `compute_loss`).

---

## Recommended `config.json` (Hindi + Mimi)

```json
{
  "huggingface": {
    "dataset_repo": "SPRINGLab/IndicVoices-R_Hindi",
    "dataset_config": null,
    "dataset_split": "train",
    "streaming": true,
    "languages": ["hi"],
    "shuffle_buffer_size": 10000,
    "min_snr": null,
    "max_cer": null,
    "val_holdout_mod": 20
  },
  "voice": {
    "strategy": "embedding_table",
    "speaker_id_source": "row",
    "max_speakers": 8192,
    "embedding_dim": 768
  },
  "model": {
    "arch": "rq_transformer_tts",
    "codebook_loss_weights": [2.0, 1.4, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
  },
  "training": {
    "device": "auto",
    "precision": "bf16",
    "batch_size": 16,
    "grad_accum_steps": 4,
    "max_steps": 6000,
    "warmup_steps": 600,
    "lr": 2e-4,
    "min_lr": 1e-5,
    "label_smoothing": 0.0,
    "num_workers": 4,
    "eval_every": 250,
    "save_every": 1000
  }
}
```

`max_steps: 6000` is the deliberate from-scratch budget (a 250M model can't
converge in 6k; this ~150M one starts forming words). Raise it only if you also
raise model size and have the compute.

---

## Pre-encoding Mimi codes (optional throughput win)

On-the-fly Mimi encoding costs GPU each step. You can pre-encode once into
`paths.mimi_code_cache_dir` (`data/mimi_codes`). Note the correct import is
`bootstrap` (there is no `setup`), and the dataset yields `wav` / `text_ids` /
`speaker_index`:

```python
# tools/precompute_mimi.py  (run once before training)
import torch
from pathlib import Path
from milli_tts.bootstrap import bootstrap
from milli_tts.data.mimi_codec import MimiCodec
from milli_tts.data.dataset import IndicVoicesDataset

bootstrap()
codec = MimiCodec.from_config(device=torch.device("cuda"))
cache_dir = Path("data/mimi_codes"); cache_dir.mkdir(parents=True, exist_ok=True)

ds = IndicVoicesDataset(register_voices=True, role="train")
for i, s in enumerate(ds):
    wav = s["wav"].unsqueeze(0).unsqueeze(0).cuda()   # [1, 1, T]
    codes = codec.encode(wav).cpu()                   # [1, Q, frames]
    torch.save({"codes": codes[0], "text_ids": s["text_ids"],
                "speaker_index": s["speaker_index"]}, cache_dir / f"{i:07d}.pt")
    if i >= 50000: break
```

This is a starting point — you still need a cache-loading dataset to consume the
`.pt` files. Not required to start training (streaming + on-GPU encode works).

---

## Metrics to watch

| Metric | Healthy @ 1k | Healthy @ 10k |
|--------|-------------|----------------|
| train/loss | 5.5–6.5 | 4.0–5.0 |
| train/acc | 3–8% | 15–30% |
| train/acc_cb0 | 1–5% | 8–20% |
| val/acc_cb0 | ~1–3% | ~6–15% |
| train/ppl | 300–600 | 60–150 |

If `train/loss` stays > 7.0 after 1k steps → LR too low / warmup too slow.
If `acc_cb0` is stuck at 0.05% past 5k steps, check, in order:
1. Data is actually flowing (the preflight + heartbeat logs print).
2. Mimi loaded the real codec (codes aren't all-zero; `codec.allow_dummy=false`).
3. Speaker conditioning isn't all-zeros (catalog built / voices_seen > 0).

---

## Launch

```bash
# Colab/Kaggle (T4+). Token + W&B key are read from config.json.
bash setup_colab.sh          # installs deps, checks dataset access, builds catalog
python train.py              # auto single/multi-GPU
python inference.py --interactive
```

---

## Architecture shapes (reference)

```
Text:           [B, Lt]     →  [B, Lt, 768]   text embedder
Audio in:       [B, 8, Ta]  →  [B, Ta, 768]   audio embedder (sum over Q)
Speaker:        [B]         →  [B, 768]        conditioner
Joint seq:      [B, Lt+Ta, 768]  → backbone (causal) → [B, Lt+Ta, 768]
h_audio:        [B, Ta, 768]     (slice Lt:)
Depth in:       [B*Ta, 8, 768]   → depth (causal over Q) → [B*Ta, 8, 2048]
Loss:           weighted CE over M×8 valid (non-padded) tokens
```

---

## Switching to the v2 model

v2 is **already registered** (`models/factory.py`), so no code edit is needed —
just set the arch in `config.json`:

```json
"model": { "arch": "rq_transformer_tts_v2" }
```

Same config keys, same `compute_loss`/`generate`, same checkpoint format as v1.
(The earlier `_REGISTRY["…"] = …` snippet was wrong — the factory uses
`@MODEL_REGISTRY.register("…")` builders.)
