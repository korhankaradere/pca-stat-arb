#!/bin/sh
# Train and evaluate the deep model for both execution lags. Safe to rerun: finished years are
# loaded from results/deep/checkpoints, so an interrupted run resumes where it stopped.
cd "$(dirname "$0")/.."
for lag in 0 1; do
  .venv/bin/python -u scripts/deep_walkforward.py --lag $lag --device mps 2>&1 \
    | grep --line-buffered -v "Data doesn't exist" | tee results/deep/deep_lag$lag.log
done
echo "Done. You can close this window."
