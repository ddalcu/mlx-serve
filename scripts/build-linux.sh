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

# ── Engines: MLX + mlx-c (+ libjinja), ds4, llama.cpp ───────────────────────
MLX_BACKEND=cuda MLX_CUDA_ARCHITECTURES="$CUDA_ARCH" ZIG="$ZIG" bash scripts/build-mlx-linux.sh
CUDA_ARCH="$CUDA_ARCH" bash scripts/build-ds4-linux.sh
bash scripts/fetch-llama.sh

# ── mlx-serve ───────────────────────────────────────────────────────────────
"$ZIG" build -Doptimize=ReleaseFast ${ZIG_CPU:+-Dcpu=$ZIG_CPU} -Dds4-commit="$(git -C lib/ds4 rev-parse --short HEAD)"
echo "== built $ROOT/zig-out/bin/mlx-serve"
