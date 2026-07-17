# KataGomo

A Hex, Y, Bent-Y and Yavalath engine and selfplay trainer based on
[KataGomo](https://github.com/hzyhhzy/KataGomo) and
[KataGo](https://github.com/lightvector/KataGo). It supports compact CNN and
transformer models, GTP analysis, batched selfplay, and PyTorch training on
macOS and Linux.

Hex connects opposite sides. Y and Bent-Y connect all three sides; Bent-Y
uses a graph board with three interior vertices of degree five. In Yavalath,
four in a row wins and exactly three loses unless the move also makes four.

## Performance

Native Metal inference on macOS uses FP16 operands for convolutions,
transformer projections, and attention, with FP32 normalization, residuals,
and outputs. Local analysis benchmarks measured **1.9× Hex B5** and
**2.3× Hex B18** search throughput compared with OpenCL. FP32 Metal alone gave
little improvement for B5; the FP16 path provided most of the gain.

Metal caches compiled graphs for each batch shape. Reusing those graphs keeps
subsequent evaluations fast; the first evaluation of a new shape includes
compilation. OpenCL is also available on macOS and is the native backend on Linux.

MPS training uses compiled FP32 forward/backward for FSON models. Warmed B5
training updates ran about **1.5× faster** than eager execution.

## Build

Run `bash scripts/build_engine.sh` for Hex, or
`bash scripts/build_engine.sh y` for the Y family. They produce
`build-engine/katago` and `build-engine-y/katago`, respectively, with runtime
libraries in each build's `native/` directory. Both use Metal and OpenCL on
macOS and OpenCL on Linux.

The build initializes the pinned KataGo inference submodule at `deps/KataGo`
on first use. Later builds reuse that revision. Set `UPSTREAM_KATAGO` to use
another source checkout explicitly.

Each executable defaults to its game family. `boardShape` selects `hex`, `y`,
`bentY`, or `yavalath` explicitly when needed.

`batchrawnn` provides batched evaluation of newline-delimited Hex positions.

## Training

The maintained recipes cycle through selfplay, replay shuffling, training,
and model export. They train across board sizes with a weighted curriculum and
support continuation from checkpoints. Warmup and learning-rate schedules
follow the checkpoint's total sample count.

Model and run choices live in [`scripts/b5/run_train.sh`](scripts/b5/run_train.sh)
and [`scripts/b5/selfplay.cfg`](scripts/b5/selfplay.cfg), with Y-family
recipes in `scripts/y-b5`, `scripts/y-b10`, `scripts/y-t4`, `scripts/by-t4`
and `scripts/yavalath-t4`. Shared launcher
operations live in [`scripts/training_common.sh`](scripts/training_common.sh);
training and optimizer code lives in [`python/`](python/).

The inherited HZY recipes are kept unchanged under `scripts/hzy/`.
