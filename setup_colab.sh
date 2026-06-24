#!/usr/bin/env bash
# One-shot Colab/T4 setup. Run from the repo root after cloning:
#   !git clone https://github.com/iNavLabsResearch/milli_tts.git
#   %cd milli_tts
#   !bash setup_colab.sh
set -e

echo "==> Installing milli_tts dependencies (T4)…"
pip -q install --upgrade pip

# Colab ships torch 2.11 + matching torchaudio.  moshi pins torch<2.10, so a plain
# `pip install -r requirements.txt` downgrades torch but leaves torchaudio on 2.11,
# which breaks with: undefined symbol: torch_library_impl
#
# Install a matched GPU torch stack first (moshi-compatible), then the rest.
CUDA_TAG=cu128
if python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null | grep -q '^12\.4'; then
  CUDA_TAG=cu124
fi
echo "    PyTorch index: ${CUDA_TAG}"
pip -q install "torch==2.9.1" "torchaudio==2.9.1" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

REQ_NO_TORCH="$(mktemp)"
grep -vE '^(torch|torchaudio)([<>=!~ \[].*)?$' requirements.txt \
  | grep -v '^#' | grep -v '^$' > "${REQ_NO_TORCH}"
pip -q install -r "${REQ_NO_TORCH}"
rm -f "${REQ_NO_TORCH}"

echo "==> Sanity import check…"
python -c "import torch, torchaudio, transformers, datasets, wandb; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

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
