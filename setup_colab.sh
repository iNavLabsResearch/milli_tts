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
