#!/usr/bin/env bash
# One-shot Linux (NVIDIA CUDA) build from a fresh checkout: checks prerequisites,
# initializes the submodules, stages Zig, MLX + mlx-c, ds4 and llama.cpp, then
# builds zig-out/bin/mlx-serve.
#
#   ./scripts/build-linux.sh
#
# CUDA_ARCH  GPU compute capabilities to build for, e.g. "120" or "75;86;120"
#            (one build for RTX 20-50). Default: the local GPUs' (nvidia-smi).
# ZIG_CPU    CPU baseline for the zig build, e.g. x86_64_v3 (CI); default native.
#            ds4 reads NATIVE_CPU_FLAG the same way (default -march=native).
# ZIG_OPTIMIZE    `zig build -Doptimize=`, default ReleaseFast (CI: Debug on a PR, never for a binary anyone runs).
# ZIG_BUILD_ARGS  Extra `zig build` arguments, space separated (CI: -Dunits=true -Dunit-groups=4 -Dstrip=true on a PR).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -d /opt/cuda/bin ]] && PATH="/opt/cuda/bin:$PATH"

# ── Prerequisites: report everything missing at once ────────────────────────
missing=()
need_cmd() { command -v "$1" >/dev/null || missing+=("$2"); }
need_file() { # description, candidate paths...
  local what="$1"; shift
  for f in "$@"; do [[ -e "$f" ]] && return; done
  missing+=("$what")
}
need_cmd git  "git"
need_cmd curl "curl"
need_cmd cmake "cmake"
need_cmd make "make"
need_cmd g++  "g++ (GCC)"
need_cmd nvcc "CUDA toolkit 13 (nvcc; Arch: cuda, Ubuntu: cuda-toolkit-13 from NVIDIA's repo)"
need_file "cuDNN (Arch: cudnn, Ubuntu: cudnn9-cuda-13)" /usr/include/cudnn.h /usr/include/x86_64-linux-gnu/cudnn.h /usr/include/aarch64-linux-gnu/cudnn.h /opt/cuda/include/cudnn.h /usr/local/cuda/include/cudnn.h
need_file "LAPACKE + OpenBLAS (Arch: lapacke openblas cblas, Ubuntu: liblapacke-dev libopenblas-dev)" /usr/include/lapacke.h /usr/local/include/lapacke.h /usr/include/openblas/lapacke.h
need_file "libwebp (Arch: libwebp, Ubuntu: libwebp-dev)" /usr/include/webp/decode.h
need_file "Avahi dns_sd compat (Arch: avahi, Ubuntu: libavahi-compat-libdnssd-dev)" /usr/include/dns_sd.h /usr/include/avahi-compat-libdns_sd/dns_sd.h
if ((${#missing[@]})); then
  echo "error: missing build prerequisites:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

if [[ -z "${CUDA_ARCH:-}" ]]; then
  command -v nvidia-smi >/dev/null && CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d . | sort -u | paste -sd';')"
  [[ -n "${CUDA_ARCH:-}" ]] || { echo "error: no NVIDIA GPU detected; set CUDA_ARCH (e.g. CUDA_ARCH=\"75;86;120\")" >&2; exit 1; }
fi
echo "== CUDA_ARCH=$CUDA_ARCH"

# ── Sources and toolchain ───────────────────────────────────────────────────
bash scripts/check-submodules.sh --fix
bash scripts/fetch-zig.sh
ZIG="$ROOT/.zig-toolchain/zig"

# ── libjinja ────────────────────────────────────────────────────────────────
# CI caches the archive and its stamp: linux.yml's key must hash every file listed here, and fetch-zig.sh.
JINJA_INPUTS=(lib/jinja_cpp/*.cpp lib/jinja_cpp/*.h lib/jinja_cpp/nlohmann/*.hpp scripts/build-linux.sh scripts/build-mlx-linux.sh)
JINJA_STAMP="lib/jinja_cpp/.linux-version"
jinja_inputs="$(sha256sum "${JINJA_INPUTS[@]}" | sha256sum | cut -c1-12)"
JINJA_WANT="zig=$("$ZIG" version) arch=$(uname -m) cpu=${ZIG_CPU:-native} inputs=$jinja_inputs"

jinja_staged() {
  [[ -f lib/jinja_cpp/libjinja-linux.a && "$(cat "$JINJA_STAMP" 2>/dev/null)" == "$JINJA_WANT" ]]
}

# build-mlx-linux.sh's loop (same flags and work directory, so the same bytes), one job per file.
build_jinja() {
  local work="$ROOT/.work-mlx-linux-cuda" f pid failed=0 pids=()
  echo "== build libjinja-linux.a (parallel)"
  mkdir -p "$work/jinja"
  for f in caps lexer parser runtime jinja_string value jinja_wrapper; do
    "$ZIG" c++ -std=c++17 -O2 -DNDEBUG ${ZIG_CPU:+-mcpu=$ZIG_CPU} -I "$ROOT/lib/jinja_cpp" \
      -c "$ROOT/lib/jinja_cpp/$f.cpp" -o "$work/jinja/$f.o" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || { echo "error: libjinja-linux.a failed to compile" >&2; exit 1; }
  rm -f "$work/libjinja-linux.a"
  ar rcsD "$work/libjinja-linux.a" "$work/jinja/"*.o
  cmp -s "$work/libjinja-linux.a" lib/jinja_cpp/libjinja-linux.a || cp "$work/libjinja-linux.a" lib/jinja_cpp/
}

# build-mlx-linux.sh's own stage check: anything it would not accept, an input override included, runs it.
mlx_staged() {
  [[ -z "${MLX_SOURCE:-}${MLX_COMMIT:-}${MLX_VERSION:-}" ]] || return 1
  local commit mlxc inputs
  commit="$(git -C lib/mlx-src rev-parse --short=12 HEAD)" || return 1
  mlxc="$(git -C lib/mlxc-src rev-parse HEAD)" || return 1
  inputs="$(cat patches/mlx*.patch scripts/build-mlx-linux.sh | sha256sum | cut -c1-12)"
  [[ -f lib/mlx/lib/libmlx.so && -f lib/mlx/lib/libmlxc.so &&
    "$(cat lib/mlx/.version 2>/dev/null)" == "mlx=$commit mlxc=$mlxc target=unknown backend=cuda cuda_arch=$CUDA_ARCH inputs=$inputs" ]]
}

# ── Engines: MLX + mlx-c (+ libjinja), ds4, llama.cpp ───────────────────────
if ! mlx_staged; then
  rm -f "$JINJA_STAMP"
  MLX_BACKEND=cuda MLX_CUDA_ARCHITECTURES="$CUDA_ARCH" ZIG="$ZIG" bash scripts/build-mlx-linux.sh
elif ! jinja_staged; then
  rm -f "$JINJA_STAMP"
  build_jinja
else
  echo "== lib/mlx and libjinja-linux.a already staged — nothing to do"
fi
echo "$JINJA_WANT" > "$JINJA_STAMP"
CUDA_ARCH="$CUDA_ARCH" bash scripts/build-ds4-linux.sh
bash scripts/fetch-llama.sh

# ── mlx-serve ───────────────────────────────────────────────────────────────
# shellcheck disable=SC2086
"$ZIG" build -Doptimize="${ZIG_OPTIMIZE:-ReleaseFast}" ${ZIG_CPU:+-Dcpu=$ZIG_CPU} -Dds4-commit="$(git -C lib/ds4 rev-parse --short HEAD)" ${ZIG_BUILD_ARGS:-}
echo "== built $ROOT/zig-out/bin/mlx-serve"
