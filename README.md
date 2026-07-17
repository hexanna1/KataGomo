# Kata-Y

This branch adapts the KataGomo Hex codebase toward the game of Y.

The intended board convention is the top-left triangular half of the Hex
rhombus. On an `N = 14` board, legal human-facing cells are `a1` through `n1`,
then `a2` through `m2`, and so on down to `a14`. In zero-indexed engine
coordinates, playable cells satisfy:

```text
x >= 0, y >= 0, and x + y < N
```

Y rules are simple: players alternately place stones, and a player wins when one
connected group touches all three sides of the triangle. There is no scoring,
capture, randomness, or strategic pass move. For compatibility with the inherited
KataGomo pipeline, pass is accepted by the engine but immediately loses.

## Building

The engine requires CMake 3.18.2 or newer, a C++14 compiler, zlib, and the
dependencies for one neural-network backend. Libzip is also required to write
self-play training data. For example, on macOS with Homebrew, build the OpenCL
backend with:

```sh
xcode-select --install
brew install cmake libzip
cmake -S cpp -B build-opencl -DUSE_BACKEND=OPENCL
cmake --build build-opencl --parallel
```

The resulting executable is `build-opencl/katago`. On other systems, select
`CUDA`, `TENSORRT`, `OPENCL`, or `EIGEN` as appropriate. See
[Compiling.md](Compiling.md) for platform and backend dependencies.

## Training

The target training loop remains the standard KataGomo flow:

```text
self-play -> shuffle training data -> train network -> export model -> repeat
```
