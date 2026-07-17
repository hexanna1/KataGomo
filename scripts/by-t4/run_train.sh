#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source ../training_common.sh
export RUN_DIR="${RUN_DIR:-$PWD/data}"
export SELFPLAY_CONFIG="$PWD/selfplay.cfg"
export BOARD_SHAPE=bentY
export MODEL_KIND=b4c256h4nbtgraph-fson-silu-rsnh
export TRAINING_NAME=by-t4 EXPORT_PREFIX=by-t4
export MAX_GAMES_TOTAL="${MAX_GAMES_TOTAL:-5000}"
export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY-}"
export MUON_WARMUP_SAMPLES="${MUON_WARMUP_SAMPLES:-250000}"
export GNORM_CLIP_SCALE="${GNORM_CLIP_SCALE:-8}"
export SWA_PERIOD_SAMPLES="${SWA_PERIOD_SAMPLES:-25000}"
export SWA_SCALE="${SWA_SCALE:-4}"
training_absolute_paths
exec ../y-t4/run_train.sh "$@"
