# Shared launcher operations; recipes own model settings and training budgets.
source "$(dirname "${BASH_SOURCE[0]}")/training_output.sh"

training_absolute_paths() {
    local name value
    for name in RUN_DIR DATA_DIR TMP_DIR ENGINE PYTHON_DIR SELFPLAY_CONFIG INITIAL_CHECKPOINT SOURCE_CHECKPOINT SEED_MODEL
    do
        value="${!name:-}"
        if [[ -n "$value" && "$value" != /* ]]; then
            export "$name=$PWD/$value"
        fi
    done
    # Bare interpreter names use PATH; interpreter paths follow the run directory.
    value="${PYTHON:-}"
    if [[ "$value" == */* && "$value" != /* ]]; then
        export PYTHON="$PWD/$value"
    fi
}

training_init() {
    SELFPLAY_CONFIG="${SELFPLAY_CONFIG:-$PWD/selfplay.cfg}"
    POS_LEN="${POS_LEN:-$(sed -nE 's/^[[:space:]]*dataBoardLen[[:space:]]*=[[:space:]]*([0-9]+).*/\1/p' "$SELFPLAY_CONFIG")}"
    training_absolute_paths
    if [[ "${TRAIN_RUN_LOCK_PID:-}" != "$$" ]]; then
        exec "${PYTHON:-python3}" "$PYTHON_DIR/run_guard.py" "$DATA_DIR" "$PWD/run_train.sh" "$@"
    fi
    mkdir -p "$DATA_DIR/selfplay" "$DATA_DIR/models" "$TMP_DIR"
    DATA_DIR_ABS="$(cd "$DATA_DIR" && pwd)"
    TMP_DIR_ABS="$(cd "$TMP_DIR" && pwd)"
}

training_lr_args() {
    LR_ARGS=(-lr-scale-auto)
    if [[ -n "$LR_SCALE_SCHEDULE" ]]; then LR_ARGS=(-lr-scale-schedule "$LR_SCALE_SCHEDULE"); fi
    if [[ -n "$LR_SCALE" ]]; then LR_ARGS=(-lr-scale "$LR_SCALE"); fi
}

training_round_budget() {
    local current
    current=$("${PYTHON:-python3}" - "$DATA_DIR_ABS/train/$TRAINING_NAME/checkpoint.ckpt" "$INITIAL_CHECKPOINT" <<'PY'
import sys, torch
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists() and sys.argv[2]:
    path = Path(sys.argv[2])
print(torch.load(path, map_location='cpu', weights_only=False)['train_state']['global_step_samples'] if path.exists() else 0)
PY
)
    ROUND_GAMES="${MAX_GAMES_TOTAL:-8000}"
    ROUND_SAMPLES="${SAMPLES_PER_EPOCH:-250000}"
    if (( current < 1000000 )); then
        ROUND_GAMES=$(((ROUND_GAMES * 2 + 4) / 5))
        ROUND_SAMPLES=$(((ROUND_SAMPLES * 2 + 4) / 5))
    fi
    ROUND_SAMPLES=$(((ROUND_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE * BATCH_SIZE))
}

# Call under the data-directory lock; the sibling temporary file shares its filesystem.
training_copy_file() (
    local source="$1" destination="$2" temporary="$2.tmp"
    trap 'rm -f "$temporary"' EXIT
    cp "$source" "$temporary" || return
    mv -f "$temporary" "$destination"
)

training_seed_from_checkpoint() {
    if [[ -n "${INITIAL_CHECKPOINT:-}" || -z "${SOURCE_CHECKPOINT:-}" ||
          -f "$DATA_DIR_ABS/train/$TRAINING_NAME/checkpoint.ckpt" ]]; then
        return 0
    fi

    INITIAL_CHECKPOINT="$DATA_DIR_ABS/seed-checkpoint.ckpt"
    if [[ ! -f "$INITIAL_CHECKPOINT" ]]; then
        training_copy_file "$SOURCE_CHECKPOINT" "$INITIAL_CHECKPOINT" || return
    fi

    (
        local model_dir="$DATA_DIR_ABS/models/seed"
        if [[ ! -f "$model_dir/model.bin.gz" ]]; then
            # The engine scans model directories recursively, so export outside them.
            local staging="$DATA_DIR_ABS/.seed-export" file
            trap 'rm -rf "$staging"' EXIT
            mkdir -p "$staging" "$model_dir" || return
            run_stage seed-export "${PYTHON:-python3}" "$PYTHON_DIR/export_model_pytorch.py" \
                -checkpoint "$INITIAL_CHECKPOINT" -use-swa \
                -export-dir "$staging" -model-name "$TRAINING_NAME-seed" -filename-prefix model || return
            gzip -f "$staging/model.bin" || return
            for file in log.txt metadata.json model.bin.gz; do
                training_copy_file "$staging/$file" "$model_dir/$file" || return
            done
        fi
        training_copy_file "$model_dir/model.bin.gz" "$DATA_DIR_ABS/latest.bin.gz"
    )
}
