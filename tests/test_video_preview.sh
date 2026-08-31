#!/usr/bin/env bash
# Latent2RGB + JPEG preview tests (issue #208).
#
# [1] Hermetic: the fixture oracle (our projection IS ComfyUI's), the JPEG
#     encoder, the filmstrip and the perceptual bar's own helpers. No MLX, no
#     model weights — runs on Linux as well as macOS.
# [2] Perceptual (opt-in, needs a VAE — NOT the transformer): a real encode →
#     decode round trip correlated against the preview, with the retired
#     golden-angle hue wheel as the control arm. Set either or both:
#       MINIMAX_H3_MODEL = pack dir holding video_vae.safetensors
#       LTX_TEST_MODEL   = dir holding vae_encoder + vae_decoder .safetensors
#       LTX_VAE_DIR      = same, when you have only the VAE pair and not the
#                          40 GB pack LTX_TEST_MODEL's other live tests demand
#     Absent = skipped, and the script still passes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x "$ROOT/.zig-toolchain/zig" ]; then
  ./scripts/fetch-zig.sh
fi
export PATH="$ROOT/.zig-toolchain:$PATH"

zig build preview-test
echo "PASS [1]: zig build preview-test"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "SKIP [2]: perceptual bar needs MLX (macOS)"
  exit 0
fi
: "${LTX_VAE_DIR:=${LTX_TEST_MODEL:-}}"
export LTX_VAE_DIR
if [ -z "${MINIMAX_H3_MODEL:-}" ] && [ -z "$LTX_VAE_DIR" ]; then
  echo "SKIP [2]: set MINIMAX_H3_MODEL and/or LTX_VAE_DIR for the perceptual bar"
  exit 0
fi

log="$(mktemp -t mlxserve-preview)"
trap 'rm -f "$log"' EXIT
zig build test -Dtest-filter="Latent2RGB preview resembles" 2>&1 | tee "$log"

# A skipped test is not a passing test: whichever pack was named must have
# produced its own correlation line, and the fit must beat the control arm.
for pair in "MINIMAX_H3_MODEL:h3-preview" "LTX_VAE_DIR:ltx-preview"; do
  var="${pair%%:*}"
  tag="${pair##*:}"
  [ -n "${!var:-}" ] || continue
  line="$(grep -o "\[$tag\].*" "$log" | tail -1 || true)"
  if [ -z "$line" ]; then
    echo "FAIL [2]: $var is set but no [$tag] line — the test skipped instead of running"
    exit 1
  fi
  echo "PASS [2]: $line"
done
