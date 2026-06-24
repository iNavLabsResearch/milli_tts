#!/usr/bin/env bash
# One-shot Colab/T4 setup. Run from the repo root after cloning:
#   !git clone https://github.com/iNavLabsResearch/milli_tts.git
#   %cd milli_tts
#   !bash setup_colab.sh
set -e

echo "==> Installing milli_tts dependencies (T4)…"
pip -q install --upgrade pip
pip -q install -r requirements.txt

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
