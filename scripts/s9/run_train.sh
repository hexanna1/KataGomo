#!/bin/bash -eu
set -eu
set -o pipefail

cd "$(dirname "$0")"

if [ -z "${ENGINE:-}" ]
then
    if [ -x ../../build-opencl/katago ]
    then
        ENGINE="../../build-opencl/katago"
    else
        ENGINE="./engine/katago"
    fi
fi

PYTHON_DIR="${PYTHON_DIR:-../../python}"
DATA_DIR="${DATA_DIR:-data}"
TMP_DIR="${TMP_DIR:-ktmp}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SHUFFLE_THREADS="${SHUFFLE_THREADS:-8}"
SHUFFLE_MIN_ROWS="${SHUFFLE_MIN_ROWS:-60000}"
MODEL_KIND="${MODEL_KIND:-b5c192nbt-fson-rvgl-bnh}"
TRAINING_NAME="${TRAINING_NAME:-s9}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-250000}"
LR_SCALE="${LR_SCALE:-2.0}"
MAX_GAMES_TOTAL="${MAX_GAMES_TOTAL:-5000}"
SKIP_SELFPLAY_ONCE="${SKIP_SELFPLAY_ONCE:-0}"
POS_LEN="${POS_LEN:-17}"
PYTHON="${PYTHON:-python3}"
DEFAULT_SEED_NAME="s6-latest"
DEFAULT_SEED_MODEL="../s6/data/latest.bin.gz"
DEFAULT_INITIAL_CHECKPOINT="../s6/data/train/s6/checkpoint.ckpt"
if [ "${SEED_MODEL+x}" != "x" ]
then
    SEED_MODEL="$DEFAULT_SEED_MODEL"
else
    SEED_MODEL="${SEED_MODEL:-}"
fi
SEED_MODEL_NAME="${SEED_MODEL_NAME:-$DEFAULT_SEED_NAME-seed}"
if [ "${INITIAL_CHECKPOINT+x}" != "x" ] && [ -f "$DEFAULT_INITIAL_CHECKPOINT" ]
then
    INITIAL_CHECKPOINT="$DEFAULT_INITIAL_CHECKPOINT"
else
    INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"
fi

mkdir -p "$DATA_DIR"/selfplay
mkdir -p "$DATA_DIR"/models
mkdir -p "$TMP_DIR"
DATA_DIR_ABS="$(cd "$DATA_DIR" && pwd)"
TMP_DIR_ABS="$(cd "$TMP_DIR" && pwd)"

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
    cp "$SEED_MODEL" "$DATA_DIR"/models/"$SEED_MODEL_NAME"/model.bin.gz
fi

while true
do
    if [ ! -x "$ENGINE" ]
    then
        echo "Engine not found or not executable: $ENGINE"
        echo "Build from the repo root with:"
        echo "  cmake -S cpp -B build-opencl -DUSE_BACKEND=OPENCL -DNO_GIT_REVISION=1"
        echo "  cmake --build build-opencl -j"
        echo "On macOS, install libzip first if needed: brew install libzip"
        echo "Or set ENGINE=/path/to/katago."
        exit 1
    fi

    if [ "$SKIP_SELFPLAY_ONCE" = "1" ]
    then
        echo "Skipping self-play once; using existing self-play data."
        SKIP_SELFPLAY_ONCE=0
    else
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$ENGINE" selfplay \
            -models-dir "$DATA_DIR"/models \
            -config selfplay.cfg \
            -output-dir "$DATA_DIR"/selfplay \
            -max-games-total "$MAX_GAMES_TOTAL"
    fi

    (
        cd "$PYTHON_DIR"
        if [ -f "$DATA_DIR_ABS"/selfplay_summary.json ]
        then
            PYTHON="$PYTHON" bash shuffle.sh "$DATA_DIR_ABS" "$TMP_DIR_ABS" "$SHUFFLE_THREADS" "$BATCH_SIZE" \
                -min-rows "$SHUFFLE_MIN_ROWS" \
                -summary-file "$DATA_DIR_ABS"/selfplay_summary.json
        else
            PYTHON="$PYTHON" bash shuffle.sh "$DATA_DIR_ABS" "$TMP_DIR_ABS" "$SHUFFLE_THREADS" "$BATCH_SIZE" \
                -min-rows "$SHUFFLE_MIN_ROWS"
        fi

        if [ ! -f "$DATA_DIR_ABS"/shuffleddata/current/train.json ]
        then
            echo "Not enough shuffled rows yet; returning to self-play."
            exit 0
        fi

        TRAIN_ARGS=(
            "$DATA_DIR_ABS" "$TRAINING_NAME" "$MODEL_KIND" "$BATCH_SIZE" main
            -lr-scale "$LR_SCALE"
            -samples-per-epoch "$SAMPLES_PER_EPOCH"
        )
        if [ -n "$INITIAL_CHECKPOINT" ]
        then
            TRAIN_ARGS+=(-initial-checkpoint "$INITIAL_CHECKPOINT")
        fi

        POS_LEN="$POS_LEN" CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" TRAIN_DEVICE="$TRAIN_DEVICE" PYTHON="$PYTHON" bash train.sh "${TRAIN_ARGS[@]}"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" PYTHON="$PYTHON" bash export.sh s9 "$DATA_DIR_ABS" 0
    )
done
