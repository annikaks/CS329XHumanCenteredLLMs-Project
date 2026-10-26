#!/usr/bin/env bash
# Train both Tinker classifiers, then score the Mistral baseline with them.
# Usage: TINKER_API_KEY=... bash scripts/run_tinker_all.sh [extra train_tinker.py args]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TINKER_API_KEY:?set TINKER_API_KEY}"

mkdir -p logs
python scripts/train_tinker.py --list-models | tee logs/tinker_models.txt
python scripts/train_tinker.py --task partner "$@" 2>&1 | tee logs/tinker_partner.log
python scripts/train_tinker.py --task rachel "$@" 2>&1 | tee logs/tinker_rachel.log
python scripts/score_generations.py --rachel-model tinker --partner-model tinker \
  --out results/classifiers/baseline_generation_scores_tinker.json
cat results/classifiers/tinker_checkpoints.json
