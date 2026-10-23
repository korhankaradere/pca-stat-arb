#!/bin/bash
# Train and evaluate the deep model for both execution lags. Safe to rerun: finished years are
# loaded from results/deep/checkpoints, so an interrupted run resumes where it stopped.
set -o pipefail
cd "$(dirname "$0")/.."
echo "running $(date)" > results/deep/status.txt
for lag in 0 1; do
  if ! .venv/bin/python -u scripts/deep_walkforward.py --lag $lag --device mps 2>&1 \
      | grep --line-buffered -v "Data doesn't exist" | tee results/deep/deep_lag$lag.log; then
    echo "Run with lag $lag failed; stopping. Rerun this script to resume."
    echo "failed lag $lag $(date)" > results/deep/status.txt
    exit 1
  fi
done
echo "done $(date)" > results/deep/status.txt
echo "Done. You can close this window."
