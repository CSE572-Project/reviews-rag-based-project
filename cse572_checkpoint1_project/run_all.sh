#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 DATA_DIR [SAMPLE_TRAIN] [SAMPLE_TEST] [MAX_FEATURES] [SEED]"
  exit 1
fi

DATA_DIR="$1"
SAMPLE_TRAIN="${2:-300000}"
SAMPLE_TEST="${3:-100000}"
MAX_FEATURES="${4:-200000}"
SEED="${5:-42}"

python -m src.run_all --data_dir "$DATA_DIR" --sample_train "$SAMPLE_TRAIN" --sample_test "$SAMPLE_TEST" --max_features "$MAX_FEATURES" --seed "$SEED"
