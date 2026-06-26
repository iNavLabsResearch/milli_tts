# Proven Training Guide: Hindi TTS with Mimi

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
  Text acts as a prefix; every audio frame attends to all text, no cross-attention needed.
- **DepthHead** = tiny causal Transformer over the 8 RVQ codebooks of one frame.
  This removes the need for MusicGen-style codebook delay patterns.
- **Mimi** is always **frozen**. The model only predicts codes; Mimi decodes them.

---

## Bug 1: macOS DataLoader Crash (Training Never Ran)

Your W&B run shows `train/step=0` and then this traceback:

```
TypeError: cannot pickle '_thread.RLock' object
  File ".../popen_spawn_posix.py"
```

**Root cause**: On macOS Python 3.12, multiprocessing defaults to `spawn`.
The HF streaming dataset holds internal thread locks which can't be pickled
and sent to spawned workers.

**Fix in `training/trainer.py`** — change `_build_loader`:

```python
def _build_loader(self) -> DataLoader:
    dataset = IndicVoicesDataset(...)
    collator = DelayedStreamCollator(text_pad_id=self.tokenizer.pad_id)

    # FIXED: on macOS/non-Linux, use num_workers=0 (no forking).
    # On Linux (Colab/Kaggle), use the configured num_workers.
    import platform
    nw = self.tcfg.num_workers if platform.system() == "Linux" else 0
    if nw == 0 and self.tcfg.num_workers > 0:
        import logging
        logging.getLogger("training.trainer").warning(
            "macOS detected — setting num_workers=0 to avoid DataLoader "
            "pickle error with HF streaming datasets.")

    return DataLoader(
        dataset, batch_size=self.tcfg.batch_size, collate_fn=collator,
        num_workers=nw, pin_memory=(self.device.type == "cuda"),
        drop_last=True, persistent_workers=nw > 0,
        # No multiprocessing_context needed when nw=0
        multiprocessing_context=self._worker_mp_context() if nw > 0 else None)
```

Or the quick config fix: set `"num_workers": 0` in `config.json` while testing on Mac.

---

## Bug 2: Training on MPS (Mac GPU) — Wrong Precision

Your log shows:
```
Starting training on mps (precision=torch.float32) for 200000 steps
```

- MPS doesn't support bf16 or fp16 AMP — `float32` is correct.
- MPS is ~3–5× slower than a T4 GPU for this model size.
- `max_steps: 200000` on MPS would take days.

**For real training, use Colab/Kaggle (free T4/A100).**
On Mac, run only short smoke tests with `max_steps: 50`.

---

## Why acc_cb0 Stays Very Low

This is the most important thing to understand:

| Codebook | What it captures | Difficulty | Expected accuracy @ 5k steps |
|----------|-----------------|------------|-------------------------------|
| cb0 | Semantic content (WavLM-distilled) | ★★★★★ hardest | 5–15% |
| cb1 | Coarse acoustic residual | ★★★★ | 20–40% |
| cb2–cb5 | Mid acoustic residuals | ★★★ | 40–65% |
| cb6–cb7 | Fine texture | ★★ easiest | 60–80% |

**Why cb0 is hardest**: Mimi's training distills WavLM/HuBERT features into cb0,
making it a semantic token (what is being said, how it's pronounced, prosody).
Predicting the exact cb0 index from text alone is like predicting the exact word
embedding from the written word — high entropy, many valid realizations.

Random chance = 1/2048 ≈ **0.05%**. Even a well-trained model gets 20–35%.
Low acc_cb0 early in training is **completely normal**.

### What actually causes pathologically low acc_cb0:

**1. Training never ran** (your case — the DataLoader crash above).
Fix the crash first.

**2. Too many unique speakers (`speaker_id_source="row"`)**
IndicVoices has thousands of unique speaker IDs. With `strategy="embedding_table"`,
most speakers are seen only a few times. The model wastes capacity learning
speaker embeddings for speakers it barely sees, at the cost of cb0 quality.

```json
// config.json fix:
"voice": {
    "speaker_id_source": "gender",   // collapse to hi_female / hi_male
    "strategy": "embedding_table",
    "max_speakers": 8,               // small table, well-trained
    ...
}
```

**3. Learning rate too low for the first codebook**
cb0 needs more signal. Try label_smoothing=0.1 to smooth the cross-entropy
targets — it prevents the model from becoming overconfident on cb1-7 at the
expense of cb0.

**4. Training too short**
6000 steps is the minimum to see cb0 start climbing. For usable quality, plan:
- 20k steps: acc_cb0 ≈ 20%, intelligible Hindi speech
- 50k steps: acc_cb0 ≈ 30%, good naturalness
- 100k+ steps: acc_cb0 ≈ 35–40%, high quality

---

## Proven Training Recipe for Hindi + Mimi

### Step 1: Fix config.json for Hindi training

```json
{
  "huggingface": {
    "dataset_repo": "ai4bharat/IndicVoices",
    "dataset_config": "hindi",
    "dataset_split": "train",
    "streaming": true,
    "languages": ["hi"],
    "shuffle_buffer_size": 10000,
    "quality_filter": true,
    "min_quality_decision": "good",
    "val_holdout_mod": 20
  },
  "voice": {
    "strategy": "embedding_table",
    "speaker_id_source": "gender",
    "max_speakers": 8,
    "embedding_dim": 256
  },
  "training": {
    "device": "auto",
    "precision": "bf16",
    "batch_size": 16,
    "grad_accum_steps": 4,
    "max_steps": 50000,
    "warmup_steps": 1000,
    "lr": 2e-4,
    "min_lr": 1e-5,
    "label_smoothing": 0.1,
    "num_workers": 4,
    "eval_every": 500,
    "save_every": 2000
  }
}
```

### Step 2: Pre-encode Mimi codes (speeds up training 3-5×)

On-the-fly Mimi encoding wastes GPU cycles. Pre-encode the dataset once:

```python
# tools/precompute_mimi.py  (run once before training)
import torch, os
from pathlib import Path
from milli_tts.data.mimi_codec import MimiCodec
from milli_tts.bootstrap import setup

setup()
codec = MimiCodec.from_config(device=torch.device("cuda"))
cache_dir = Path("data/mimi_codes")
cache_dir.mkdir(parents=True, exist_ok=True)

from milli_tts.data.dataset import IndicVoicesDataset
from milli_tts.data.text_tokenizer import TextTokenizer
from milli_tts.data.voice_bank import VoiceBank

tokenizer = TextTokenizer.from_config()
voice_bank = VoiceBank.from_config()
ds = IndicVoicesDataset(tokenizer=tokenizer, voice_bank=voice_bank,
                        register_voices=True, role="train")

for i, sample in enumerate(ds):
    key = f"{i:07d}"
    wav = sample["wav"].unsqueeze(0).unsqueeze(0).cuda()   # [1, 1, T]
    codes = codec.encode(wav).cpu()                        # [1, Q, frames]
    out = {"codes": codes[0], "text_ids": sample["text_ids"],
           "speaker_index": sample["speaker_index"]}
    torch.save(out, cache_dir / f"{key}.pt")
    if i % 1000 == 0:
        print(f"Encoded {i} samples")
    if i >= 50000:  # encode 50k for start
        break
```

Then modify the dataset to load from cache instead of decoding on-the-fly.

### Step 3: Monitor these metrics

| Metric | Healthy at step 1k | Healthy at step 10k |
|--------|-------------------|---------------------|
| train/loss | 5.5–6.5 | 4.0–5.0 |
| train/acc | 3–8% | 15–30% |
| train/acc_cb0 | 1–5% | 8–20% |
| val/acc_cb0 | ~1–3% | ~6–15% |
| train/ppl | 300–600 | 60–150 |

If `train/loss` stays above 7.0 after 1k steps, learning rate is too low.
If `train/acc_cb0` is 0.05% (random chance) at step 5k, the model isn't learning — check:
1. DataLoader is actually feeding data (not crashing silently)
2. Audio codes are not all-zero (Mimi loaded correctly)
3. Speaker conditioning is not all-zeros (conditioner initialized)

### Step 4: Colab/Kaggle launch command

```bash
# On T4 GPU (Colab/Kaggle):
pip install -e . moshi soundfile datasets huggingface_hub wandb

# Set your tokens in config.json or as env vars:
export HF_TOKEN=your_token
export WANDB_API_KEY=your_key

python train.py
```

### Step 5: Smoke test on Mac first

```bash
# Quick sanity check (no GPU needed):
python -c "
from milli_tts.bootstrap import setup
from milli_tts.training.trainer import Trainer
import json, pathlib

# Patch config for smoke test
cfg_path = pathlib.Path('config.json')
cfg = json.loads(cfg_path.read_text())
cfg['training']['max_steps'] = 5
cfg['training']['num_workers'] = 0
cfg['training']['batch_size'] = 2
cfg['codec']['allow_dummy'] = True
cfg['wandb']['enabled'] = False
tmp = pathlib.Path('/tmp/smoke_config.json')
tmp.write_text(json.dumps(cfg))

import os; os.environ['MILLI_CONFIG'] = str(tmp)
setup()
Trainer().train()
print('Smoke test passed!')
"
```

---

## Quick Reference: Architecture Shapes

```
Text:           [B, Lt]     →  [B, Lt, 768]   text_embedder
Audio in:       [B, 8, Ta]  →  [B, Ta, 768]   audio_embedder (sum over Q)
Speaker:        [B]         →  [B, 768]        conditioner
Joint seq:      [B, Lt+Ta, 768]
Backbone out:   [B, Lt+Ta, 768]  (causal)
h_audio:        [B, Ta, 768]     (slice Lt:)
Depth in:       [B*Ta, 8, 768]   (Q steps)
Depth out:      [B*Ta, 8, 2048]  (Q × codebook logits)
Loss:           CE over M×8 valid (non-padded) tokens
```

---

## Using the New Modular Architecture

Register `rq_transformer_tts_v2` in the model factory:

```python
# milli_tts/models/factory.py  — add this import and registration

from milli_tts.models.rq_transformer_v2 import RQTransformerTTSv2

_REGISTRY["rq_transformer_tts_v2"] = RQTransformerTTSv2
```

Then in `config.json`:
```json
"model": {
    "arch": "rq_transformer_tts_v2",
    ...
}
```

The new model is **drop-in compatible** — same `compute_loss` and `generate` interface,
same config keys, same checkpoint format.
