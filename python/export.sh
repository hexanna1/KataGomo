#!/bin/bash -eu
set -eu
set -o pipefail
{
# Export PyTorch checkpoints to engine models, using averaged weights.
# Main exports go to modelstobetested/ or models/; extra exports go to models_extra/.

if [[ $# -ne 3 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR USEGATING"
    echo "Run from the python directory containing export_model_pytorch.py."
    echo "NAMEPREFIX string prefix for this training run, try to pick something globally unique. Will be displayed to users when KataGo loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "USEGATING = 1 to use gatekeeper, 0 to not use gatekeeper and output directly to models/"
    exit 0
fi
NAMEPREFIX="$1"
shift
BASEDIR="$1"
shift
USEGATING="$1"
shift
PYTHON="${PYTHON:-python3}"

#------------------------------------------------------------------------------

mkdir -p "$BASEDIR"/torchmodels_toexport
mkdir -p "$BASEDIR"/torchmodels_toexport_extra
mkdir -p "$BASEDIR"/modelstobetested
mkdir -p "$BASEDIR"/models_extra
mkdir -p "$BASEDIR"/models

function exportStuff() {
    FROMDIR="$1"
    TODIR="$2"

    #Sort by timestamp so that we process in order of oldest to newest if there are multiple
    while IFS= read -r FILEPATH
    do
        if [ -z "$FILEPATH" ]
        then
            continue
        fi
        # Skip checkpoint directories while the trainer is publishing them.
        if [[ "${FILEPATH: -4}" == ".tmp" ]]
        then
            echo "Skipping tmp file:" "$FILEPATH"
        elif [[ "${FILEPATH: -9}" == ".exported" ]]
        then
            echo "Skipping self tmp file:" "$FILEPATH"
        else
            echo "Found model to export:" "$FILEPATH"
            NAME="$(basename "$FILEPATH")"

            SRC="$BASEDIR"/"$FROMDIR"/"$NAME"
            TMPDST="$BASEDIR"/"$FROMDIR"/"$NAME".exported
            TARGET="$BASEDIR"/"$TODIR"/"$NAME"

            if [ -d "$BASEDIR"/modelstobetested/"$NAME" ] ||  \
               [ -d "$BASEDIR"/rejectedmodels/"$NAME" ] || \
               [ -d "$BASEDIR"/models/"$NAME" ] || \
               [ -d "$BASEDIR"/models_extra/"$NAME" ] || \
               [ -d "$BASEDIR"/modelsuploaded/"$NAME" ]
            then
                echo "Model with same name aleady exists, so skipping:" "$SRC"
            else
                rm -rf "$TMPDST"
                mkdir "$TMPDST"

                set -x
                "$PYTHON" ./export_model_pytorch.py \
                        -checkpoint "$SRC"/model.ckpt \
                        -export-dir "$TMPDST" \
                        -model-name "$NAMEPREFIX""-""$NAME" \
                        -filename-prefix model \
                        -use-swa
                set +x

                # Keep the queued checkpoint until publication without copying its optimizer state.
                ln "$SRC"/model.ckpt "$TMPDST"/model.ckpt
                gzip "$TMPDST"/model.bin

                #Make a bunch of the directories that selfplay will need so that there isn't a race on the selfplay
                #machines to concurrently make it, since sometimes concurrent making of the same directory can corrupt
                #a filesystem
                #Only when not gating. When gating, gatekeeper is responsible.
                if [ "$USEGATING" -eq 0 ]
                then
                    if [ "$TODIR" != "models_extra" ]
                    then
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"/sgfs
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"/tdata
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"/vdata
                        cp -f "$TMPDST"/model.bin.gz "$BASEDIR"/.latest.bin.gz
                        mv -f "$BASEDIR"/.latest.bin.gz "$BASEDIR"/latest.bin.gz
                    fi
                fi

                mv "$TMPDST" "$TARGET"
                rm -r "$SRC"
                echo "Done exporting:" "$NAME" "to" "$TARGET"
            fi
        fi
    done < <("$PYTHON" - "$BASEDIR"/"$FROMDIR" <<'PY'
import os
import sys

base = sys.argv[1]
if not os.path.isdir(base):
    sys.exit(0)

paths = [os.path.join(base, name) for name in os.listdir(base)]
paths.sort(key=lambda path: os.path.getmtime(path))
for path in paths:
    print(path)
PY
    )
}

if [ "$USEGATING" -eq 0 ]
then
    exportStuff "torchmodels_toexport" "models"
else
    exportStuff "torchmodels_toexport" "modelstobetested"
fi
exportStuff "torchmodels_toexport_extra" "models_extra"

exit 0
}
