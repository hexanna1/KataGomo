# KataGomo

KataGomo adapts KataGo's engine and reinforcement-learning pipeline to board
games other than Go. Game implementations live on separate branches, which may
also include ready-to-run training configurations under `scripts/`.

Game branches:

- [Y2026](https://github.com/hexanna1/KataGomo/tree/Y2026)
- [Hex2026](https://github.com/hexanna1/KataGomo/tree/Hex2026)
- [Quax2026](https://github.com/hexanna1/KataGomo/tree/Quax2026)

## Compiling

Check out the branch for the game you want to use. The engine requires CMake
3.18.2 or newer, a C++14 compiler, zlib, libzip, and the dependencies for one
neural-network backend. For example, on macOS with Homebrew, build the OpenCL
backend with:

```sh
xcode-select --install
brew install cmake libzip
cmake -S cpp -B build-opencl -DUSE_BACKEND=OPENCL
cmake --build build-opencl --parallel
```

The executable is `build-opencl/katago`. Select `CUDA`, `TENSORRT`, `OPENCL`, or
`EIGEN` as appropriate for your system.

## Training an existing game

Choose an example under `scripts/` close to the board size and model you want.
Review `selfplay.cfg`, `gtp.local.cfg`, and the variables near the top of
`run_train.sh`, then run:

```sh
cd scripts/<run>
./run_train.sh
```

It is often more efficient to train first on a smaller board and initialize a
larger-board run from that checkpoint. If the features and network architecture
are compatible, set `INITIAL_CHECKPOINT` and recheck board-size-dependent
settings.

## Adding a game

Start from a recent branch with similar rules and understand the roles of
`Board`, `BoardHistory`, and the neural-network inputs before changing them. A
working move generator is not enough: search, self-play, and training retain
assumptions inherited from other games.

The easy-to-miss integration points are:

- board geometry and valid symmetries in both C++ inference and Python data
  augmentation;
- capture, ownership, and move-domination assumptions in search helpers and
  neural-network targets;
- random openings that expose the network to pass-like transitions when an
  internal pass action is retained, while self-play treats voluntary passes as
  losses so the policy does not learn them as normal moves;
- hashes and history for repetition or other state not visible in the stones.

Use existing game branches to find every integration point rather than assuming
that changing only the core board rules is sufficient.
