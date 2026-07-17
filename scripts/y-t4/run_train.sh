#!/bin/bash -eu
set -euo pipefail

cd "$(dirname "$0")"
source ../training_common.sh

ENGINE="${ENGINE:-../../build-engine-y/katago}"
RUN_DIR="${RUN_DIR:-data}"
BOARD_SHAPE="${BOARD_SHAPE:-y}"
ROUNDS="${ROUNDS:-0}"
SELFPLAY_OVERRIDES="${SELFPLAY_OVERRIDES:-boardShape=$BOARD_SHAPE}"
PYTHON_DIR="${PYTHON_DIR:-../../python}"
DATA_DIR="${DATA_DIR:-$RUN_DIR}"
TMP_DIR="${TMP_DIR:-$DATA_DIR/ktmp}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SHUFFLE_THREADS="${SHUFFLE_THREADS:-8}"
SHUFFLE_MIN_ROWS="${SHUFFLE_MIN_ROWS:-60000}"
SHUFFLE_EXPAND_WINDOW_PER_ROW="${SHUFFLE_EXPAND_WINDOW_PER_ROW:-0.4}"
MODEL_KIND="${MODEL_KIND:-b4c256h4nbttflrs-y}"
TRAINING_NAME="${TRAINING_NAME:-y11-t4nbt}"
EXPORT_PREFIX="${EXPORT_PREFIX:-y-t4nbt}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"
LR_SCALE="${LR_SCALE:-}"
LR_SCALE_SCHEDULE="${LR_SCALE_SCHEDULE:-}"
training_lr_args
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY-}"
MUON_WARMUP_SAMPLES="${MUON_WARMUP_SAMPLES:-250000}"
GNORM_CLIP_SCALE="${GNORM_CLIP_SCALE:-8}"
SWA_PERIOD_SAMPLES="${SWA_PERIOD_SAMPLES:-25000}"
SWA_SCALE="${SWA_SCALE:-4}"
MAX_GAMES_TOTAL="${MAX_GAMES_TOTAL:-}"
SKIP_SELFPLAY_ONCE="${SKIP_SELFPLAY_ONCE:-0}"
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

training_seed_from_checkpoint

round=0
while (( ROUNDS == 0 || round < ROUNDS ))
do
    round=$((round + 1))
    training_round_budget

    if [ ! -x "$ENGINE" ]
    then
        echo "Engine not found or not executable: $ENGINE"
        echo "Set ENGINE=/path/to/katago, or run ../build_engine.sh."
        exit 1
    fi
    if [ "$SKIP_SELFPLAY_ONCE" = "1" ]
    then
        echo "Skipping self-play once; using existing self-play data."
        SKIP_SELFPLAY_ONCE=0
    else
        run_stage selfplay env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$ENGINE" selfplay \
            -models-dir "$DATA_DIR"/models \
            -config "$SELFPLAY_CONFIG" -override-config "$SELFPLAY_OVERRIDES" \
            -output-dir "$DATA_DIR"/selfplay \
            -max-games-total "$ROUND_GAMES"
    fi

    (
        cd "$PYTHON_DIR"
        if [ -f "$DATA_DIR_ABS"/selfplay_summary.json ]
        then
            run_stage shuffle bash shuffle.sh "$DATA_DIR_ABS" "$TMP_DIR_ABS" "$SHUFFLE_THREADS" "$BATCH_SIZE" \
                -expand-window-per-row "$SHUFFLE_EXPAND_WINDOW_PER_ROW" -min-rows "$SHUFFLE_MIN_ROWS" \
                -summary-file "$DATA_DIR_ABS"/selfplay_summary.json
        else
            run_stage shuffle bash shuffle.sh "$DATA_DIR_ABS" "$TMP_DIR_ABS" "$SHUFFLE_THREADS" "$BATCH_SIZE" \
                -expand-window-per-row "$SHUFFLE_EXPAND_WINDOW_PER_ROW" -min-rows "$SHUFFLE_MIN_ROWS"
        fi

        if [ ! -f "$DATA_DIR_ABS"/shuffleddata/current/train.json ]
        then
            echo "Not enough shuffled rows yet; returning to self-play."
            exit 0
        fi

        TRAIN_ARGS=(
            "$DATA_DIR_ABS" "$TRAINING_NAME" "$MODEL_KIND" "$BATCH_SIZE" main
            -use-muon
            "${LR_ARGS[@]}"
            -muon-warmup-samples "$MUON_WARMUP_SAMPLES"
            -gnorm-clip-scale "$GNORM_CLIP_SCALE"
            -swa-period-samples "$SWA_PERIOD_SAMPLES"
            -swa-scale "$SWA_SCALE"
            -samples-per-epoch "$ROUND_SAMPLES"
            -quit-if-no-data
        )
        if [ -n "$MUON_WEIGHT_DECAY" ]; then
            TRAIN_ARGS+=(-muon-weight-decay "$MUON_WEIGHT_DECAY")
        fi
        if [ -n "$INITIAL_CHECKPOINT" ]
        then
            TRAIN_ARGS+=(-initial-checkpoint "$INITIAL_CHECKPOINT")
        fi
        run_stage train env POS_LEN="$POS_LEN" BOARD_SHAPE="$BOARD_SHAPE" CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" TRAIN_DEVICE="$TRAIN_DEVICE" bash train.sh "${TRAIN_ARGS[@]}"
        run_stage export env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" bash export.sh "$EXPORT_PREFIX" "$DATA_DIR_ABS" 0
    )
done
