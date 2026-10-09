#!/usr/bin/env bash
# Stage the ds4 engine (lib/ds4, CUDA backend) for the Linux serve build as
# lib/ds4-linux/libds4.a. On macOS build.zig compiles ds4 (Metal) itself; the
# CUDA backend needs nvcc, so here upstream's own Makefile builds its library
# objects (its CORE_OBJS) and we archive them.
#
# Usage: [CUDA_ARCH=120] ./scripts/build-ds4-linux.sh
# CUDA_ARCH defaults to the first GPU's compute capability (nvidia-smi). A list
# ("75;86;120") builds one fat archive, SASS + PTX per arch, without the
# Blackwell-only MXFP4 path (ds4 enables it build-wide, not per arch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS4="$ROOT/lib/ds4"
STAGE="$ROOT/lib/ds4-linux"

[[ -f "$DS4/ds4.c" ]] || { echo "error: lib/ds4 is missing. Run: git submodule update --init lib/ds4" >&2; exit 1; }
[[ -d /opt/cuda/bin ]] && PATH="/opt/cuda/bin:$PATH"
command -v nvcc >/dev/null || { echo "error: nvcc not on PATH (install the CUDA toolkit)" >&2; exit 1; }
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}"
IFS=';' read -ra ARCHS <<< "${CUDA_ARCH//sm_/}"

if ((${#ARCHS[@]} == 1)); then
  MAKE_ARGS=(-C "$DS4" CUDA_ARCH="sm_${ARCHS[0]}")
else
  GENCODE=""
  for a in "${ARCHS[@]}"; do
    # Blackwell's block-scaled MMA in the vendored llama.cpp kernels needs the arch-specific target.
    [[ $a == 12[01] ]] && a+=a
    GENCODE+=" -gencode arch=compute_$a,code=[sm_$a,compute_$a]"
  done
  MAKE_ARGS=(-C "$DS4" NVCC_ARCH_FLAGS="$GENCODE")
fi
OBJS="$(make -s "${MAKE_ARGS[@]}" --eval='print-core-objs: ; @echo $(CORE_OBJS)' print-core-objs)"
echo "== ds4 ($(git -C "$DS4" rev-parse --short HEAD)) CUDA_ARCH=$CUDA_ARCH"
# The stamp names every input, so a matching archive (a CI cache hit) skips the build. The
# objects build in-tree and make tracks sources, not flags: rebuild when the targets change.
FLAVOR="$(git -C "$DS4" rev-parse HEAD) $CUDA_ARCH ${NATIVE_CPU_FLAG:-native} $(sha256sum "${BASH_SOURCE[0]}" | cut -c1-12)"
if [[ "$(cat "$STAGE/.flavor" 2>/dev/null)" == "$FLAVOR" ]]; then
  [[ -f "$STAGE/libds4.a" ]] && { echo "== lib/ds4-linux/libds4.a already staged — nothing to do"; exit 0; }
else
  MAKE_ARGS+=(-B)
fi
# shellcheck disable=SC2086
make -j"$(nproc)" "${MAKE_ARGS[@]}" $OBJS

mkdir -p "$STAGE"
rm -f "$STAGE/libds4.a"
# shellcheck disable=SC2086
(cd "$DS4" && ar rcs "$STAGE/libds4.a" $OBJS)
echo "$FLAVOR" > "$STAGE/.flavor"
echo "== staged $STAGE/libds4.a"
