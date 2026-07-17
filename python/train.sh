#!/bin/bash -eu
set -eu
set -o pipefail
{
#Runs tensorflow training in $BASEDIR/train/$TRAININGNAME
#Should be run once per persistent training process.
#Outputs results in torchmodels_toexport/ in an ongoing basis (EXPORTMODE == "main").
#Or, to torchmodels_toexport_extra/ (EXPORTMODE == "extra").
#Or just trains without exporting (EXPORTMODE == "trainonly").

if [[ $# -lt 5 ]]
then
    echo "Usage: $0 BASEDIR TRAININGNAME MODELKIND BATCHSIZE EXPORTMODE OTHERARGS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TRANINGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train, like b10c128, see ../modelconfigs.py"
    echo "BATCHSIZE number of samples to concat together per batch for training, must match shuffle"
    echo "EXPORTMODE 'main': train and export for selfplay. 'extra': train and export extra non-selfplay model. 'trainonly': train without export"
    exit 0
fi
BASEDIR="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift
BATCHSIZE="$1"
shift
EXPORTMODE="$1"
shift
POS_LEN="${POS_LEN:-15}"
BOARD_SHAPE="${BOARD_SHAPE:-y}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
USE_FP16="${USE_FP16:-auto}"
PYTHON="${PYTHON:-python3}"

GITROOTDIR="$(git rev-parse --show-toplevel)"

#------------------------------------------------------------------------------
set -x

mkdir -p "$BASEDIR"/train/"$TRAININGNAME"

# For archival and logging purposes - you can look back and see exactly the python code on a particular date
DATE_FOR_FILENAME=$(date "+%Y%m%d-%H%M%S")
DATED_ARCHIVE="$BASEDIR"/scripts/train/dated/"$DATE_FOR_FILENAME"
mkdir -p "$DATED_ARCHIVE"
cp "$GITROOTDIR"/python/*.py "$GITROOTDIR"/python/train.sh "$DATED_ARCHIVE"


if [ "$EXPORTMODE" == "main" ]
then
    EXPORT_SUBDIR=torchmodels_toexport
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "extra" ]
then
    EXPORT_SUBDIR=torchmodels_toexport_extra
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "trainonly" ]
then
    EXPORT_SUBDIR=torchmodels_toexport_extra
    EXTRAFLAG="-no-export"
else
    echo "EXPORTMODE was not 'main' or 'extra' or 'trainonly', run with no arguments for usage"
    exit 1
fi

if [ "$USE_FP16" == "auto" ]
then
    if [ "$TRAIN_DEVICE" == "cuda" ]
    then
        USE_FP16=1
    elif [ "$TRAIN_DEVICE" == "auto" ] && "$PYTHON" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
        USE_FP16=1
    else
        USE_FP16=0
    fi
fi

DEVICE_ARGS=(-device "$TRAIN_DEVICE")
case "$USE_FP16" in
    1|true|TRUE|yes|YES)
        DEVICE_ARGS+=(-use-fp16)
        ;;
    0|false|FALSE|no|NO)
        ;;
    *)
        echo "USE_FP16 must be auto, 1, or 0"
        exit 1
        ;;
esac

   # -main-loss-scale 1.0 \
    #-intermediate-loss-scale 0.0 \
time "$PYTHON" ./train.py \
     -traindir "$BASEDIR"/train/"$TRAININGNAME" \
     -datadir "$BASEDIR"/shuffleddata/current/ \
     -exportdir "$BASEDIR"/"$EXPORT_SUBDIR" \
     -exportprefix "$TRAININGNAME" \
     -pos-len "$POS_LEN" \
     -board-shape "$BOARD_SHAPE" \
     -batch-size "$BATCHSIZE" \
     -model-kind "$MODELKIND" \
     -max-epochs-this-instance 1 \
     -lr-scale 1\
     -samples-per-epoch 2000000 \
    -soft-policy-weight-scale 8.0 \
    -value-loss-scale 0.6 \
    -td-value-loss-scales 0.6,0.6,0.6 \
     -lookahead-alpha 0.5 \
     -lookahead-k 6 \
     -swa-scale 1.0 \
     "${DEVICE_ARGS[@]}" \
     $EXTRAFLAG \
     "$@" \
     2>&1 | tee -a "$BASEDIR"/train/"$TRAININGNAME"/stdout.txt

exit 0
}
