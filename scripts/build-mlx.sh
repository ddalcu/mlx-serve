#!/usr/bin/env bash
# Build the pinned mlx + mlx-c submodules (lib/mlx-src, lib/mlxc-src) into
# lib/mlx/{lib,include} with MLX's NAX (M5 neural-accelerator) kernels ENABLED.
#
# Why not Homebrew: the brew bottle is compiled at MACOSX_DEPLOYMENT_TARGET
# 26.0 (the Tahoe builders' point release), which fails MLX's CMake gate
# (SDK >= 26.2 AND deployment target >= 26.2) — the NAX kernels are silently
# skipped and MLX_METAL_NO_NAX hard-wires is_nax_available() to false, even on
# M5 hardware. Building ourselves at 26.2 flips the gate; the runtime check in
# MLX still gates dispatch on GPU gen >= 17, so M1–M4 behavior is unchanged.
#
# The gate fails SILENTLY upstream (a configure-time warning only landed after
# v0.32.0), so this script ASSERTS the result: no *_nax kernels in the built
# metallib = hard failure, never a quietly-degraded stage.
#
# Consequence of the 26.2 deployment target: the staged dylibs (and anything
# bundling them) require macOS >= 26.2 at runtime.
#
# This is the single source of truth for the pinned mlx/mlx-c versions: the
# submodule SHAs. Bump by checking out a new tag in the submodule; CI and
# local builds rebuild automatically (stamp mismatch). Guard test:
# tests/test_mlx_staged_nax.sh.
set -euo pipefail

DEPLOYMENT_TARGET="${MLX_DEPLOYMENT_TARGET:-26.2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MLX_SRC="$REPO_ROOT/lib/mlx-src"
MLXC_SRC="$REPO_ROOT/lib/mlxc-src"
STAGE="$REPO_ROOT/lib/mlx"
BUILD_ROOT="$REPO_ROOT/lib/.mlx-build"
STAMP="$STAGE/.version"

die() { echo "[build-mlx] ERROR: $*" >&2; exit 1; }

[ -f "$MLX_SRC/CMakeLists.txt" ] && [ -f "$MLXC_SRC/CMakeLists.txt" ] \
  || die "submodules missing — run: git submodule update --init lib/mlx-src lib/mlxc-src"

MLX_SHA="$(git -C "$MLX_SRC" rev-parse --short=12 HEAD)"
MLXC_SHA="$(git -C "$MLXC_SRC" rev-parse --short=12 HEAD)"
# The mlx-c patch below is part of the build: an edited patch must rebuild.
MLXC_PATCH="$REPO_ROOT/patches/mlxc-gather-qmm-global-scale.patch"
PATCH_SHA="$(shasum "$MLXC_PATCH" | cut -c1-12)"
WANT="mlx=$MLX_SHA mlxc=$MLXC_SHA patch=$PATCH_SHA target=$DEPLOYMENT_TARGET"

# Idempotent: skip when the staged build already matches the pinned SHAs.
if [ -f "$STAMP" ] && [ -f "$STAGE/lib/libmlx.dylib" ] \
   && [ -f "$STAGE/lib/libmlxc.dylib" ] && [ -f "$STAGE/lib/mlx.metallib" ]; then
  if [ "$(cat "$STAMP")" = "$WANT" ]; then
    echo "[build-mlx] lib/mlx already at ($WANT) — nothing to do"
    exit 0
  fi
  echo "[build-mlx] staged '$(cat "$STAMP")' != '$WANT' — rebuilding"
fi

# ── Toolchain preflight: every one of these failing would otherwise surface
# as a silently NAX-less metallib, not an error. ──────────────────────────────
xcrun -sdk macosx metal --version >/dev/null 2>&1 \
  || die "Metal compiler unavailable — Xcode 26 ships it as a separate download: xcodebuild -downloadComponent MetalToolchain"

SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
SDK_MAJOR="${SDK_VERSION%%.*}"
SDK_MINOR="$(echo "$SDK_VERSION" | cut -d. -f2)"
if [ "$SDK_MAJOR" -lt 26 ] || { [ "$SDK_MAJOR" -eq 26 ] && [ "${SDK_MINOR:-0}" -lt 2 ]; }; then
  die "macOS SDK $SDK_VERSION < 26.2 — MLX's NAX kernel gate needs SDK >= 26.2 (update Xcode)"
fi
echo "[build-mlx] SDK $SDK_VERSION, deployment target $DEPLOYMENT_TARGET, $WANT"

NCPU="$(sysctl -n hw.ncpu)"

# ── mlx (C++ core + metallib) ────────────────────────────────────────────────
cmake -S "$MLX_SRC" -B "$BUILD_ROOT/mlx" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$STAGE"
cmake --build "$BUILD_ROOT/mlx" -j "$NCPU"
cmake --install "$BUILD_ROOT/mlx" >/dev/null

# ── mlx-c against the staged mlx (same pairing brew uses: USE_SYSTEM_MLX) ────
# MLX 0.32.3's gather_qmm takes a global_scale before sorted_indices; the
# pinned mlx-c predates it (the same patch the Linux build applies). The files
# the patch touches are reset first, so an edited patch replaces the old one;
# local edits anywhere else in the submodule are kept.
git -C "$MLXC_SRC" checkout -- $(git -C "$MLXC_SRC" apply --numstat -p1 "$MLXC_PATCH" | cut -f3)
git -C "$MLXC_SRC" apply -p1 "$MLXC_PATCH"
cmake -S "$MLXC_SRC" -B "$BUILD_ROOT/mlxc" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_USE_SYSTEM_MLX=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$STAGE" \
  -DCMAKE_INSTALL_PREFIX="$STAGE"
cmake --build "$BUILD_ROOT/mlxc" -j "$NCPU"
cmake --install "$BUILD_ROOT/mlxc" >/dev/null

# ── Assert the point of the exercise ─────────────────────────────────────────
METALLIB="$STAGE/lib/mlx.metallib"
[ -f "$METALLIB" ] || die "mlx.metallib not at $METALLIB — mlx changed its install layout"

NAX_COUNT="$(strings "$METALLIB" | grep -c "_nax" || true)"
[ "$NAX_COUNT" -gt 0 ] \
  || die "built metallib contains ZERO *_nax kernels — the 26.2 gate failed silently (check SDK/deployment target/Metal Toolchain)"

MINOS="$(otool -l "$STAGE/lib/libmlx.dylib" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')"
case "$MINOS" in
  26.[2-9]*|2[7-9]*|[3-9]*) ;;
  *) die "libmlx.dylib minos '$MINOS' < 26.2 — deployment target did not take" ;;
esac

[ -f "$STAGE/lib/libmlxc.dylib" ] || die "libmlxc.dylib missing from stage"

echo "$WANT" > "$STAMP"
echo "[build-mlx] staged lib/mlx OK: $NAX_COUNT NAX symbol hits, minos $MINOS"
