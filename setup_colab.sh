#!/usr/bin/env bash
# One-shot Colab/T4 setup. Run from the repo root after cloning:
#   !git clone https://github.com/iNavLabsResearch/milli_tts.git
#   %cd milli_tts
#   !bash setup_colab.sh
set -e

echo "==> Installing milli_tts dependencies (T4)…"
pip -q install --upgrade pip

# FFmpeg/libsndfile so soundfile + librosa can decode IndicVoices audio without
# torchcodec (which often fails on Colab: libavutil.so.* / torch ABI mismatch).
echo "==> Installing audio codecs (ffmpeg, libsndfile)…"
(apt-get -qq update && apt-get -qq install -y ffmpeg libsndfile1) >/dev/null 2>&1 \
  || echo "    (apt install skipped — soundfile usually still works)"

# Colab ships torch 2.11 + matching torchaudio.  moshi pins torch<2.10, so a plain
# `pip install -r requirements.txt` downgrades torch but leaves torchaudio on 2.11,
# which breaks with: undefined symbol: torch_library_impl
#
# Fix: install torch + torchaudio *together* from one CUDA index with a single
# `<2.10` constraint, so pip picks a co-released, ABI-matched pair whose wheels
# actually exist on that index (no brittle hard-coded version).
#
# Picking the CUDA index:
#   * Blackwell GPUs (sm_100/sm_120 — RTX 50-series, RTX PRO 6000, B200) need
#     cu128 wheels (torch>=2.7). cu124 wheels only ship kernels up to sm_90 and
#     crash with "no kernel image is available for execution on the device".
#   * We detect the *live* GPU's compute capability via nvidia-smi first, because
#     on a fresh box the installed torch may not know its CUDA version yet
#     (chicken-and-egg). compute_cap is like "12.0" (Blackwell) / "9.0" (Hopper).
CUDA_TAG=cu124
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -n1 | tr -d ' ')"
CC_MAJOR="${CC%%.*}"
if [ -n "${CC_MAJOR}" ] && [ "${CC_MAJOR}" -ge 10 ] 2>/dev/null; then
  # sm_100 / sm_120 -> Blackwell -> must use cu128.
  CUDA_TAG=cu128
else
  # Fall back to whatever CUDA the currently-installed torch was built against.
  DETECTED="$(python -c 'import torch; print((torch.version.cuda or "").replace(".",""))' 2>/dev/null || true)"
  case "${DETECTED}" in
    129|128|127|126) CUDA_TAG=cu128 ;;
    124|123|122|121) CUDA_TAG=cu124 ;;
  esac
fi
echo "    PyTorch index: ${CUDA_TAG} (gpu compute_cap='${CC:-?}')"
if ! pip -q install "torch<2.10" "torchaudio<2.10" \
      --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"; then
  # Retry the same index once (transient network), then — only for non-Blackwell —
  # fall back to cu124. Never silently downgrade a Blackwell box to cu124: that
  # reinstalls a wheel its GPU can't run.
  echo "    install on ${CUDA_TAG} failed; retrying…"
  FALLBACK_TAG="${CUDA_TAG}"
  [ "${CUDA_TAG}" != "cu128" ] && FALLBACK_TAG=cu124
  pip -q install "torch<2.10" "torchaudio<2.10" \
    --index-url "https://download.pytorch.org/whl/${FALLBACK_TAG}"
fi

REQ_NO_TORCH="$(mktemp)"
grep -vE '^(torch|torchaudio)([<>=!~ \[].*)?$' requirements.txt \
  | grep -v '^#' | grep -v '^$' > "${REQ_NO_TORCH}"
pip -q install -r "${REQ_NO_TORCH}"
rm -f "${REQ_NO_TORCH}"

echo "==> Installing milli_tts package…"
pip -q install -e .

echo "==> Sanity import check…"
python -c "import milli_tts.data; import torch, transformers, datasets, wandb; \
print('milli_tts.data OK'); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# torchaudio is optional (librosa/soundfile are used as fallbacks); never fail setup on it.
python -c "import torchaudio; print('torchaudio', torchaudio.__version__, 'OK')" \
  || echo '    WARN: torchaudio failed to load; continuing (librosa/soundfile fallback active).'

# `datasets` will try to decode the `audio` column with torchcodec, whose prebuilt
# libs are ABI-pinned to a specific torch+FFmpeg combo that Kaggle/Colab rarely
# matches — importing it HARD-CRASHES the interpreter (libavutil.so.* missing /
# torch CUDA symbol mismatch). This project never needs it: audio is decoded with
# soundfile/librosa and the dataset casts Audio(decode=False). Remove it so
# `datasets` can't touch it. (No-op if absent.)
echo "==> Removing torchcodec (incompatible on T4; we decode via soundfile)…"
pip -q uninstall -y torchcodec >/dev/null 2>&1 || true

# ── Dataset access check (repo + token come from config.json) ────────────────
# Hindi-ONLY. Everything — the dataset repo, its config, and the HF token — is
# read from config.json via bootstrap() (which exports HF_TOKEN from the
# REV-encoded token in the file). So this verifies EXACTLY what training will
# stream (currently SPRINGLab/IndicVoices-R_Hindi) with no hardcoded repo and no
# separate HF_TOKEN export needed. PREFETCH_HINDI=1 pre-downloads the repo.
echo "==> Verifying dataset access (repo + token from config.json)…"
python - <<'PY' || echo "    WARN: dataset access check failed — confirm huggingface.token in config.json is valid and, if the repo is gated, that you've accepted its terms on the Hub."
import os
from milli_tts.bootstrap import bootstrap
from milli_tts.core.static_memory_cache import StaticMemoryCache
bootstrap()                                    # loads config.json, exports HF_TOKEN
cfg = StaticMemoryCache.config()
repo, dscfg = cfg.huggingface.dataset_repo, cfg.huggingface.dataset_config
split, tok = cfg.huggingface.dataset_split, os.environ.get("HF_TOKEN")
from datasets import load_dataset
def _open(c):
    if c:
        return load_dataset(repo, c, split=split, streaming=True, token=tok)
    return load_dataset(repo, split=split, streaming=True, token=tok)
print("    Dataset:", repo, "| config:", repr(dscfg), "| token:",
      "present" if tok else "MISSING")
try:
    ds = _open(dscfg)
except Exception as e:
    print("    (config", repr(dscfg), "unavailable:", str(e)[:120],
          "— retrying with no config)")
    ds = _open(None)
# Read the audio column as raw bytes (decode=False) — exactly like the training
# dataset — so iterating never invokes torchcodec.
from datasets import Audio
feats = getattr(ds, "features", None) or {}
for ac in ("audio", "audio_filepath", "wav"):
    if ac in feats:
        try:
            ds = ds.cast_column(ac, Audio(decode=False))
        except Exception:
            pass
        break
row = next(iter(ds))
print("    OK — stream is live. First-row columns:", list(row.keys())[:14])
PY

if [ "${PREFETCH_HINDI:-0}" = "1" ]; then
  echo "==> PREFETCH_HINDI=1 -> downloading the dataset snapshot to the HF cache…"
  python - <<'PY' || echo "    WARN: prefetch failed (training still works via streaming)."
import os
from milli_tts.bootstrap import bootstrap
from milli_tts.core.static_memory_cache import StaticMemoryCache
bootstrap()
cfg = StaticMemoryCache.config()
from huggingface_hub import snapshot_download
# IndicVoices-R_Hindi is already a single Hindi repo, so grab it whole.
path = snapshot_download(repo_id=cfg.huggingface.dataset_repo, repo_type="dataset",
                         token=os.environ.get("HF_TOKEN"))
print("    Data cached at:", path)
PY
else
  echo "    (streaming during training; set PREFETCH_HINDI=1 to pre-download)"
fi

# Build the "perfect dataset": the per-speaker catalog (collision-free voice
# indices, so inference can pick any Hindi speaker_id) + the deterministic
# train/val split sizes. Reads metadata only — no audio decode. Capped by
# CATALOG_MAX_ROWS (default 30000) to keep setup snappy; run
# `python tools/build_dataset.py --max-rows 0` for a full pass.
echo "==> Building Hindi speaker catalog + train/val split…"
python tools/build_dataset.py --max-rows "${CATALOG_MAX_ROWS:-30000}" \
  || echo "    WARN: catalog build skipped/failed — training still works (speaker indices fall back to hashing)."

echo "==> Running CPU smoke test (dummy codec, tiny model)…"
python -m tests.smoke_test || echo 'smoke test skipped/failed (non-fatal)'

cat <<'EOF'

Setup done. Next:
  1. Put your secrets in the environment (or create a .env file):
       import os
       os.environ["HF_TOKEN"]       = "hf_..."
       os.environ["WANDB_API_KEY"]  = "..."
  2. Edit config.json (dataset_config language, batch_size, max_steps).
  3. Train:     python train.py
  4. Inference: python inference.py --interactive
EOF
