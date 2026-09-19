#!/usr/bin/env bash
# Stage the Linux mlx + mlx-c pair into lib/mlx/ for the Linux serve build
# (build.zig addLinuxServe). The macOS counterpart is scripts/build-mlx.sh.
#
# mlx comes from the Linux fork tree (mlx-omarchy: upstream MLX + the
# Omarchy Honeykrisp Vulkan backend overlay + patches), prepared ONCE with:
#
#   cd <mlx-omarchy checkout> && scripts/prepare-mlx.sh
#   # prints e.g. <mlx-omarchy>/.work/mlx
#
# mlx-c is the pinned submodule lib/mlxc-src (56b2d39), built against the
# fork's installed headers with MLX_C_USE_SYSTEM_MLX=ON — exactly the binding
# target the macOS build stages from the same submodule.
#
# Usage:
#   MLX_SOURCE=<staged tree> ./scripts/build-mlx-linux.sh
#   MLX_SOURCE=<staged tree> MLX_BUILD_DIR=<existing cmake build> ./scripts/build-mlx-linux.sh
#
# MLX_BUILD_DIR (optional) points at a cmake build dir of MLX_SOURCE (e.g. the
# wheel build's mlx.core dir); the just-built `mlx` target is then installed
# from it instead of recompiling. MLX_COMMIT / MLX_VERSION (optional) are
# stamped into lib/mlx/.version for `mlx-serve --version`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/.work-mlx-linux"
STAGE="$ROOT/lib/mlx"

MLX_SOURCE="${MLX_SOURCE:?set MLX_SOURCE to the staged Linux mlx tree (mlx-omarchy scripts/prepare-mlx.sh)}"
MLX_COMMIT="${MLX_COMMIT:-unknown}"
MLX_VERSION="${MLX_VERSION:-unknown}"

[[ -f "$MLX_SOURCE/CMakeLists.txt" ]] || { echo "error: $MLX_SOURCE is not an mlx source tree" >&2; exit 1; }
MLXC_SRC="$ROOT/lib/mlxc-src"
[[ -d "$MLXC_SRC/mlx/c" ]] || {
  echo "error: lib/mlxc-src is missing. Run: git submodule update --init lib/mlxc-src" >&2
  exit 1
}

ZIG="${ZIG:-zig}"
command -v "$ZIG" >/dev/null || { echo "error: zig not on PATH (set ZIG=...)" >&2; exit 1; }
command -v cmake >/dev/null || { echo "error: cmake not on PATH" >&2; exit 1; }
mkdir -p "$WORK"

# ── 1. Build + install libmlx (shared, omarchy Vulkan backend) ──────────
if [[ -n "${MLX_BUILD_DIR:-}" ]]; then
  echo "== install mlx from existing build: $MLX_BUILD_DIR"
  cmake --install "$MLX_BUILD_DIR" --prefix "$WORK/mlx-prefix" >/dev/null
else
  echo "== configure mlx: $MLX_SOURCE"
  cmake -S "$MLX_SOURCE" -B "$WORK/mlx-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DMLX_BUILD_METAL=OFF \
    -DMLX_BUILD_CPU=ON \
    -DMLX_BUILD_CUDA=OFF \
    -DMLX_BUILD_OMARCHY=ON \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF
  cmake --build "$WORK/mlx-build" --parallel "$(nproc)"
  cmake --install "$WORK/mlx-build" --prefix "$WORK/mlx-prefix" >/dev/null
fi

# ── 2. Build mlx-c against the fork's installed MLX ─────────────────────
echo "== patch mlx-c for the fork's gather_qmm signature"
if grep -q 'global_scale: no C ABI surface yet' "$MLXC_SRC/mlx/c/ops.cpp"; then
  echo "   gather_qmm: already applied"
else
  git -C "$MLXC_SRC" apply -p1 "$ROOT/patches/mlxc-gather-qmm-global-scale.patch"
fi
if grep -q 'no baseline _Float16' "$MLXC_SRC/mlx/c/half.h"; then
  echo "   f16 alias: already applied"
else
  git -C "$MLXC_SRC" apply -p1 "$ROOT/patches/mlxc-x86-f16-data-alias.patch"
fi

echo "== configure mlx-c: $MLXC_SRC"
# $ORIGIN so libmlxc.so finds the staged libmlx.so sitting next to it; a
# build-tree RUNPATH would point back into $WORK and break after staging.
# The fork's exported `mlx` target carries an INTERFACE link on Vulkan::Headers
# (the omarchy backend), so the consumer project must find_package(Vulkan)
# BEFORE find_package(MLX) — injected right after mlx-c's project() via
# CMAKE_PROJECT_INCLUDE (TOP_LEVEL_INCLUDES runs before project(), where the
# system search paths don't exist yet and find_package fails silently).
cat > "$WORK/vulkan-preload.cmake" <<'EOF'
find_package(Vulkan)
EOF
cmake -S "$MLXC_SRC" -B "$WORK/mlxc-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_USE_SYSTEM_MLX=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$WORK/mlx-prefix" \
  -DCMAKE_PROJECT_INCLUDE="$WORK/vulkan-preload.cmake" \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
cmake --build "$WORK/mlxc-build" --target mlxc --parallel "$(nproc)"
cmake --install "$WORK/mlxc-build" --prefix "$WORK/mlx-prefix" >/dev/null

# ── 3. Stage into lib/mlx (the layout addMlxLib expects) ────────────────
echo "== stage $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE/include" "$STAGE/lib"
cp -a "$WORK/mlx-prefix/include/." "$STAGE/include/"
find "$WORK/mlx-prefix/lib" -maxdepth 1 -name 'libmlx.so*' -exec cp -a {} "$STAGE/lib/" \;
find "$WORK/mlx-prefix/lib" -maxdepth 1 -name 'libmlxc.so*' -exec cp -a {} "$STAGE/lib/" \;
printf 'mlx=%s mlxc=%s target=%s\n' "$MLX_COMMIT" \
  "$(git -C "$MLXC_SRC" rev-parse HEAD)" "$MLX_VERSION" > "$STAGE/.version"

# ── 4. libjinja for Linux (ELF objects; the committed libjinja.a is Mach-O) ──
echo "== build lib/jinja_cpp/libjinja-linux.a"
mkdir -p "$WORK/jinja"
for f in caps lexer parser runtime jinja_string value jinja_wrapper; do
  "$ZIG" c++ -std=c++17 -O2 -DNDEBUG -I "$ROOT/lib/jinja_cpp" \
    -c "$ROOT/lib/jinja_cpp/$f.cpp" -o "$WORK/jinja/$f.o"
done
rm -f "$ROOT/lib/jinja_cpp/libjinja-linux.a"
ar rcs "$ROOT/lib/jinja_cpp/libjinja-linux.a" "$WORK/jinja/"*.o

echo "== done: $STAGE"
ls -l "$STAGE/lib" "$STAGE/.version"
