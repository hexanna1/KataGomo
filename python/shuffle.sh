#!/bin/bash -eu
set -eu
set -o pipefail
{
#Shuffles and copies selfplay training from selfplay/ to shuffleddata/current/
#Should be run periodically.

if [[ $# -lt 4 ]]
then
    echo "Usage: $0 BASEDIR TMPDIR NTHREADS BATCHSIZE"
    echo "Currently expects to be run from within the 'python' directory of the KataGo repo, or otherwise in the same dir as shuffle.py."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TMPDIR scratch space, ideally on fast local disk, unique to this loop"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    echo "BATCHSIZE number of samples to concat together per batch for training"
    exit 0
fi
BASEDIR="$1"
shift
TMPDIR="$1"
shift
NTHREADS="$1"
shift
BATCHSIZE="$1"
shift
PYTHON="${PYTHON:-python3}"

#------------------------------------------------------------------------------

OUTDIR=$(date "+%Y%m%d-%H%M%S")
OUTDIRTRAIN=$OUTDIR/train
OUTDIRVAL=$OUTDIR/val

mkdir -p "$BASEDIR"/shuffleddata/$OUTDIR
mkdir -p "$TMPDIR"/train
mkdir -p "$TMPDIR"/val

echo "Beginning shuffle at" $(date "+%Y-%m-%d %H:%M:%S")

#set -x
(
    time "$PYTHON" ./shuffle.py \
         "$BASEDIR"/selfplay/ \
         -expand-window-per-row 0.4 \
         -taper-window-exponent 0.65 \
         -out-dir "$BASEDIR"/shuffleddata/$OUTDIRTRAIN \
         -out-tmp-dir "$TMPDIR"/train \
         -approx-rows-per-out-file 70000 \
         -num-processes "$NTHREADS" \
         -batch-size "$BATCHSIZE" \
         -min-rows 60000 \
         -keep-target-rows 2100000 \
         -only-include-md5-path-prop-lbound 0.00 \
         -only-include-md5-path-prop-ubound 0.95 \
         -output-npz \
         "$@" \
         2>&1 | tee "$BASEDIR"/shuffleddata/$OUTDIR/outtrain.txt
)
(
    time "$PYTHON" ./shuffle.py \
         "$BASEDIR"/selfplay/ \
         -expand-window-per-row 0.4 \
         -taper-window-exponent 0.65 \
         -out-dir "$BASEDIR"/shuffleddata/$OUTDIRVAL \
         -out-tmp-dir "$TMPDIR"/val \
         -approx-rows-per-out-file 70000 \
         -num-processes "$NTHREADS" \
         -batch-size "$BATCHSIZE" \
         -min-rows 60000 \
         -keep-target-rows 51200 \
         -only-include-md5-path-prop-lbound 0.95 \
         -only-include-md5-path-prop-ubound 1.00 \
         -output-npz \
         "$@" \
         2>&1 | tee "$BASEDIR"/shuffleddata/$OUTDIR/outval.txt
)

#set +x

#Just in case, give a little time for nfs
sleep 10

#rm if it already exists

rm -rf "$BASEDIR"/shuffleddata/current_tmp
ln -s "$OUTDIR" "$BASEDIR"/shuffleddata/current_tmp
rm -rf "$BASEDIR"/shuffleddata/current
mv "$BASEDIR"/shuffleddata/current_tmp "$BASEDIR"/shuffleddata/current



# CLEANUP ---------------------------------------------------------------

#Among shuffled dirs older than 2 hours, remove all but the most recent 5 of them.
#This should be VERY conservative and allow plenty of time for the training to switch
#to newer ones as they get generated.
echo "Cleaning up any old dirs"
"$PYTHON" - "$BASEDIR"/shuffleddata <<'PY'
import os
import shutil
import sys
import time

base = sys.argv[1]
cutoff = time.time() - 120 * 60
old_dirs = []
for name in os.listdir(base):
    path = os.path.join(base, name)
    if os.path.islink(path) or not os.path.isdir(path):
        continue
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        continue
    if mtime < cutoff:
        old_dirs.append((name, path))

old_dirs.sort()
for _, path in old_dirs[:-5]:
    shutil.rmtree(path)
PY

echo "Finished shuffle at" $(date "+%Y-%m-%d %H:%M:%S")
#Make a little space between shuffles
echo ""
echo ""

exit 0
}
