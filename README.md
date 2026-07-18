# KataGomo

KataGomo adapts KataGo's engine and reinforcement-learning pipeline to board
games other than Go. Hex, Y, Bent-Y and Yavalath share one implementation,
with maintained training configurations under `scripts/`.

[Engine source and training recipes](https://github.com/hexanna1/KataGomo/tree/games)

## Compiling

Check out the engine source. The maintained build uses CMake 3.21+, C++17,
zlib, libzip and OpenCL development libraries. On macOS it also uses Swift
5.9+ and Python 3 with the `ninja` package. Git initializes the pinned
[KataGo](https://github.com/lightvector/KataGo) inference submodule on first build:

```sh
bash scripts/build_engine.sh
```

`UPSTREAM_KATAGO` selects another upstream checkout. The Hex executable is
`build-engine/katago`. Use `bash scripts/build_engine.sh y` for the Y-family
executable at `build-engine-y/katago`. Runtime libraries live in each build’s
`native/` directory. macOS uses
Metal by default; set `KATAGOMO_INFERENCE_BACKEND=opencl` to select OpenCL.
Linux uses OpenCL. Other CMake backends remain available.

For the released size-24 Y model, build with
`COMPILE_MAX_BOARD_LEN=24 bash scripts/build_engine.sh y`.

## Training an existing game

Choose a maintained run under `scripts/` close to the board size and model you
want. Original HZY recipes are preserved unchanged under `scripts/hzy/`.
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

Start from a game implementation with similar rules and understand the roles of
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

Use the existing game implementations to find every integration point rather than assuming
that changing only the core board rules is sufficient.
