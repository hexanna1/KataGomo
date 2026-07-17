#!/bin/bash -eu
set -eu
set -o pipefail

cd "$(dirname "$0")"
source ../training_common.sh

ENGINE="${ENGINE:-../../build-engine/katago}"

PYTHON_DIR="${PYTHON_DIR:-../../python}"
DATA_DIR="${DATA_DIR:-data}"
TMP_DIR="${TMP_DIR:-$DATA_DIR/ktmp}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SHUFFLE_THREADS="${SHUFFLE_THREADS:-8}"
SHUFFLE_MIN_ROWS="${SHUFFLE_MIN_ROWS:-60000}"
SHUFFLE_EXPAND_WINDOW_PER_ROW="${SHUFFLE_EXPAND_WINDOW_PER_ROW:-0.4}"
BOARD_SHAPE=hex
MODEL_KIND="${MODEL_KIND:-b5c192nbt-fson-rvgl-bnh}"
TRAINING_NAME="${TRAINING_NAME:-b5nbt-14}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"
LR_SCALE="${LR_SCALE:-}"
LR_SCALE_SCHEDULE="${LR_SCALE_SCHEDULE:-}"
training_lr_args
LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-250000}"
MAX_GAMES_TOTAL="${MAX_GAMES_TOTAL:-5000}"
SKIP_SELFPLAY_ONCE="${SKIP_SELFPLAY_ONCE:-0}"
PYTHON="${PYTHON:-python3}"
SEED_MODEL="${SEED_MODEL:-}"
SEED_MODEL_NAME="${SEED_MODEL_NAME:-$TRAINING_NAME-seed}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"

training_init "$@"

if [ -n "$INITIAL_CHECKPOINT" ]
then
    if [ ! -f "$INITIAL_CHECKPOINT" ]
    then
        echo "Initial checkpoint not found: $INITIAL_CHECKPOINT"
        exit 1
    fi
    INITIAL_CHECKPOINT="$(cd "$(dirname "$INITIAL_CHECKPOINT")" && pwd)/$(basename "$INITIAL_CHECKPOINT")"
fi

if [ -f "$SEED_MODEL" ] && [ ! -f "$DATA_DIR"/models/"$SEED_MODEL_NAME"/model.bin.gz ]
then
    mkdir -p "$DATA_DIR"/models/"$SEED_MODEL_NAME"
    training_copy_file "$SEED_MODEL" "$DATA_DIR"/models/"$SEED_MODEL_NAME"/model.bin.gz
fi

while true
do
    training_round_budget
    if [ ! -x "$ENGINE" ]
    then
        echo "Engine not found or not executable: $ENGINE"
        echo "Build from the repo root with:"
        echo "  bash scripts/build_engine.sh"
        echo "On macOS, install libzip first if needed: brew install libzip"
        echo "Or set ENGINE=/path/to/katago."
        exit 1
    fi

    if [ "$SKIP_SELFPLAY_ONCE" = "1" ]
    then
        echo "Skipping self-play once; using existing self-play data."
        SKIP_SELFPLAY_ONCE=0
    else
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" run_stage selfplay "$ENGINE" selfplay \
            -models-dir "$DATA_DIR"/models \
            -config "$SELFPLAY_CONFIG" \
            -output-dir "$DATA_DIR"/selfplay \
            -max-games-total "$ROUND_GAMES"
    fi

    (
        cd "$PYTHON_DIR"
        if [ -f "$DATA_DIR_ABS"/selfplay_summary.json ]
        then
            PYTHON="$PYTHON" run_stage shuffle bash shuffle.sh "$DATA_DIR_ABS" "$TMP_DIR_ABS" "$SHUFFLE_THREADS" "$BATCH_SIZE" \
                -expand-window-per-row "$SHUFFLE_EXPAND_WINDOW_PER_ROW" \
                -min-rows "$SHUFFLE_MIN_ROWS" \
                -summary-file "$DATA_DIR_ABS"/selfplay_summary.json
        else
            PYTHON="$PYTHON" run_stage shuffle bash shuffle.sh "$DATA_DIR_ABS" "$TMP_DIR_ABS" "$SHUFFLE_THREADS" "$BATCH_SIZE" \
                -expand-window-per-row "$SHUFFLE_EXPAND_WINDOW_PER_ROW" \
                -min-rows "$SHUFFLE_MIN_ROWS"
        fi

        if [ ! -f "$DATA_DIR_ABS"/shuffleddata/current/train.json ]
        then
            echo "Not enough shuffled rows yet; returning to self-play."
            exit 0
        fi

        TRAIN_ARGS=(
            "$DATA_DIR_ABS" "$TRAINING_NAME" "$MODEL_KIND" "$BATCH_SIZE" main
            "${LR_ARGS[@]}"
            -lr-warmup-samples "$LR_WARMUP_SAMPLES"
            -samples-per-epoch "$ROUND_SAMPLES"
            -quit-if-no-data
        )
        if [ -n "$INITIAL_CHECKPOINT" ]
        then
            TRAIN_ARGS+=(-initial-checkpoint "$INITIAL_CHECKPOINT")
        fi

        BOARD_SHAPE="$BOARD_SHAPE" POS_LEN="$POS_LEN" CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" TRAIN_DEVICE="$TRAIN_DEVICE" PYTHON="$PYTHON" run_stage train bash train.sh "${TRAIN_ARGS[@]}"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" PYTHON="$PYTHON" run_stage export bash export.sh b5nbt-14 "$DATA_DIR_ABS" 0
    )
done
