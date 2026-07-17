#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
SOURCE="$ROOT/cpp/neuralnet/native"
deps="${NATIVE_DEPS_PREFIX:-$ROOT/deps/native}"
export PKG_CONFIG_PATH="$deps/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
game="${1:-hex}"
case "$game" in
    hex) default_len=19; game_suffix=""; cmake_game=HEX ;;
    y) default_len=21; game_suffix="-y"; cmake_game=Y ;;
    *) echo "Usage: $0 [hex|y]" >&2; exit 1 ;;
esac
if [[ -z "${UPSTREAM_KATAGO:-}" ]]; then
    git -C "$ROOT" submodule update --init --checkout --depth 1 -- deps/KataGo
    UPSTREAM_KATAGO="$ROOT/deps/KataGo"
fi
if [[ "$game" == y && -f "$ROOT/build-engine-y/CMakeCache.txt" ]]; then
    default_len="$(awk -F= '$1 == "COMPILE_MAX_BOARD_LEN:STRING" {print $2}' "$ROOT/build-engine-y/CMakeCache.txt")"
fi
board_len="${COMPILE_MAX_BOARD_LEN:-$default_len}"
build_suffix=""
if [[ "$game" == hex && "$board_len" != "$default_len" ]]; then
    build_suffix="-$board_len"
fi
engine="$ROOT/build-engine$game_suffix$build_suffix"
native="$engine/native"
dispatch="$native/dispatch"
args=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$deps" -DPKG_CONFIG_ARGN=--static
      -DUSE_BIGGER_BOARDS_EXPENSIVE=OFF -DCOMPILE_MAX_BOARD_LEN="$board_len")
backends=(OPENCL)
suffix=so
if [[ "$(uname)" == Darwin ]]; then
    backends+=(METAL)
    suffix=dylib
fi
mkdir -p "$native"
cmake -S "$ROOT/cpp" -B "$engine" "${args[@]}" -DDEFAULT_GAME="$cmake_game" -DUSE_BACKEND=NATIVE \
    -DNATIVE_LIBRARY="$dispatch/libkatagoinference.$suffix" \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$engine/bin" -DNO_GIT_REVISION=1 \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="$native"
for backend in "${backends[@]}"; do
    backend_lower="$(echo "$backend" | tr '[:upper:]' '[:lower:]')"
    backend_dir="$native/$backend_lower"
    backend_args=("${args[@]}" -DUPSTREAM_KATAGO="$UPSTREAM_KATAGO" -DUSE_BACKEND="$backend")
    if [[ "$backend" == METAL ]]; then
        backend_args+=(-G Ninja -DCMAKE_MAKE_PROGRAM="$("${PYTHON:-python3}" -c 'import ninja; print(ninja.BIN_DIR + "/ninja")')")
    fi
    cmake -S "$SOURCE" -B "$backend_dir" "${backend_args[@]}"
    cmake --build "$backend_dir" --target katagoinference -j "${BUILD_JOBS:-4}"
done
cmake -S "$SOURCE" -B "$dispatch" "${args[@]}" -DUPSTREAM_KATAGO="$UPSTREAM_KATAGO" -DUSE_BACKEND=DISPATCH
cmake --build "$dispatch" --target katagoinference -j "${BUILD_JOBS:-4}"
cmake --build "$engine" --target katago -j "${BUILD_JOBS:-4}"

# Prepare every runtime file before publishing any of them.
for backend in "${backends[@]}"; do
    backend_lower="$(echo "$backend" | tr '[:upper:]' '[:lower:]')"
    cp "$native/$backend_lower/libkatagoinference_$backend_lower.$suffix" "$native/.libkatagoinference_$backend_lower.$suffix"
done
cp "$dispatch/libkatagoinference.$suffix" "$native/.libkatagoinference.$suffix"
cp "$engine/bin/katago" "$engine/.katago"
# Rename into place so running engines retain their mapped files.
for backend in "${backends[@]}"; do
    backend_lower="$(echo "$backend" | tr '[:upper:]' '[:lower:]')"
    mv -f "$native/.libkatagoinference_$backend_lower.$suffix" "$native/libkatagoinference_$backend_lower.$suffix"
done
mv -f "$native/.libkatagoinference.$suffix" "$native/libkatagoinference.$suffix"
mv -f "$engine/.katago" "$engine/katago"
