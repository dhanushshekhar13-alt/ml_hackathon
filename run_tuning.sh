#!/bin/bash
# Bayesian tuning (LightGBM) + HMM decoder tuning + final selection and route comparison. ~10 min on Laptop 1.
# Usage:  bash run_tuning.sh
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
cd src
echo "== [1/3] Bayesian optimisation (Optuna TPE, 30 trials)";  python tune_bayes.py 30
echo "== [2/3] HMM decoder tuning on training trips";         python tune_hmm.py
echo "== [3/3] Model selection + classifier vs HMM";          python evaluate_hmm.py
cd ..
echo ""
echo "== DONE. Restart the API (Ctrl+C, then bash start_api.sh) to serve the tuned model + decoder."
