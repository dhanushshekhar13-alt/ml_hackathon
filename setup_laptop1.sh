#!/bin/bash
# Laptop 1 one-shot setup: virtualenv, dependencies, dataset build, EDA, training, HMM evaluation.
# Usage:  bash setup_laptop1.sh
set -e
cd "$(dirname "$0")"
echo "== Python: $(python3 --version)"
if [ ! -d "../Student folder/train" ] && [ -z "$PATHSNAP_DATA" ]; then
  echo "!! Dataset not found at ../Student folder (set PATHSNAP_DATA to its path)"; exit 1
fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt
cd src
echo "== [1/4] Building candidate/feature tables";  python build_dataset.py
echo "== [2/4] EDA plots and Folium maps";           python eda.py
echo "== [3/4] Training 5 models (about 5 minutes)"; python train_models.py
echo "== [4/4] Model selection + HMM evaluation";    python evaluate_hmm.py
cd ..
echo ""
echo "== SETUP COMPLETE. Next:  bash start_api.sh"
