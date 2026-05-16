#!/usr/bin/env bash
# Local GPU-server full run (replaces SLURM nibi_job.sh).
# Runs ONE seed; for paper-grade results call this 3 times with SEED=0,1,2.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/local_env.sh

mkdir -p logs

RUN_ID="${RUN_ID:-run_local_seed${SEED:-0}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MODELS="${MODELS:-distilbert-base-uncased bert-base-uncased distilgpt2 gpt2}"
K_VALUES="${K_VALUES:-1 2 3 5 8 10 15 20}"
WINOBIAS_TYPES="${WINOBIAS_TYPES:-type_1 type_2}"
UTILITY_DATASETS="${UTILITY_DATASETS:-wikitext2}"
MAX_PAIRS_WINOBIAS="${MAX_PAIRS_WINOBIAS:-0}"
MAX_PAIRS_STEREOSET="${MAX_PAIRS_STEREOSET:-0}"
EVAL_MAX_PAIRS_WINOBIAS="${EVAL_MAX_PAIRS_WINOBIAS:-0}"
EVAL_MAX_PAIRS_STEREOSET="${EVAL_MAX_PAIRS_STEREOSET:-0}"
UTILITY_MAX_TEXTS="${UTILITY_MAX_TEXTS:-2000}"
MAX_LENGTH="${MAX_LENGTH:-128}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
FAIRETELL_N_DIRS="${FAIRETELL_N_DIRS:-4}"
FAIRETELL_DIR_MAX_PAIRS="${FAIRETELL_DIR_MAX_PAIRS:-400}"

read -r -a MODEL_LIST <<< "$MODELS"
read -r -a K_LIST <<< "$K_VALUES"
read -r -a WB_TYPE_LIST <<< "$WINOBIAS_TYPES"
read -r -a UTIL_DS_LIST <<< "$UTILITY_DATASETS"

echo "== ROBIN local run =="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Project: $PROJECT_ROOT"
echo "RUN_ID=$RUN_ID  SEED=$SEED  DEVICE=$DEVICE  BATCH_SIZE=$BATCH_SIZE"
echo "HF_HOME=$HF_HOME"

"$VENV_PY" -c "import torch; print(f'[run] torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

CMD=(
  "$VENV_PY" -m robin.run_all
  --outdir outputs
  --run_id "$RUN_ID"
  --device "$DEVICE"
  --models "${MODEL_LIST[@]}"
  --max_length "$MAX_LENGTH"
  --max_pairs_winobias "$MAX_PAIRS_WINOBIAS"
  --max_pairs_stereoset "$MAX_PAIRS_STEREOSET"
  --eval_max_pairs_winobias "$EVAL_MAX_PAIRS_WINOBIAS"
  --eval_max_pairs_stereoset "$EVAL_MAX_PAIRS_STEREOSET"
  --utility_max_texts "$UTILITY_MAX_TEXTS"
  --utility_batch_size "$BATCH_SIZE"
  --utility_datasets "${UTIL_DS_LIST[@]}"
  --k_values "${K_LIST[@]}"
  --winobias_types "${WB_TYPE_LIST[@]}"
  --seed "$SEED"
  --fairetell_n_dirs "$FAIRETELL_N_DIRS"
  --fairetell_dir_max_pairs "$FAIRETELL_DIR_MAX_PAIRS"
)

printf '[run] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "End: $(date)"
echo "Outputs: $PROJECT_ROOT/outputs/$RUN_ID"
