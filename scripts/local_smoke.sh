#!/usr/bin/env bash
# Local GPU-server smoke test (replaces scripts/smoke_test.sh which assumes .venv).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# shellcheck disable=SC1091
source scripts/local_env.sh

RUN_ID="${RUN_ID:-smoke_distilbert}"
MODEL="${MODEL:-distilbert-base-uncased}"
DEVICE="${DEVICE:-cuda}"

"$VENV_PY" -m robin.run_all \
  --outdir outputs \
  --run_id "$RUN_ID" \
  --device "$DEVICE" \
  --models "$MODEL" \
  --max_pairs_winobias 2 \
  --max_pairs_stereoset 2 \
  --eval_max_pairs_winobias 2 \
  --eval_max_pairs_stereoset 2 \
  --utility_max_texts 4 \
  --utility_batch_size 2 \
  --utility_datasets wikitext2 wikitext103 \
  --winobias_types type_1 type_2 \
  --k_values 1 2 \
  --seed 0

test -f "outputs/$RUN_ID/GLOBAL_SUMMARY.json"
test -f "outputs/$RUN_ID/$MODEL/eval/ablation_results.json"
test -f "outputs/$RUN_ID/$MODEL/rankings/winobias_rankings.json"
test -f "outputs/$RUN_ID/$MODEL/rankings/stereoset_rankings.json"
echo "[smoke] Passed. Outputs: $PROJECT_ROOT/outputs/$RUN_ID"
