#!/usr/bin/env bash
# Run all 3 seeds back-to-back on the local GPU. Logs go to logs/.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

for SEED in 0 1 2; do
  RUN_ID="full_seed${SEED}"
  LOG="logs/${RUN_ID}.log"
  echo "==> SEED=$SEED RUN_ID=$RUN_ID  (log: $LOG)"
  RUN_ID="$RUN_ID" SEED="$SEED" bash scripts/local_run.sh 2>&1 | tee "$LOG"
done

echo "All 3 seeds complete."
