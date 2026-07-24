#!/usr/bin/env bash
# CPU-only Qwen preprocessing parity against the pinned mlx-vlm processor.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${1:-tests/fixtures/house.jpeg}"
FIXTURE="${QWEN_PREPROCESS_FIXTURE:-/tmp/qwen_preprocess_fixture}"
PYTHON="${PYTHON:-python3}"
ZIG="${ZIG:-zig}"
MIN_PIXELS="${QWEN_PREPROCESS_MIN_PIXELS:-3136}"
MAX_PIXELS="${QWEN_PREPROCESS_MAX_PIXELS:-1003520}"
DOWNSCALE_FIXTURE="${QWEN_PREPROCESS_DOWNSCALE_FIXTURE:-${FIXTURE}_downscale}"
DOWNSCALE_MIN_PIXELS="${QWEN_PREPROCESS_DOWNSCALE_MIN_PIXELS:-$MIN_PIXELS}"
DOWNSCALE_MAX_PIXELS="${QWEN_PREPROCESS_DOWNSCALE_MAX_PIXELS:-65536}"

run_parity() {
  local fixture="$1"
  local min_pixels="$2"
  local max_pixels="$3"

  "$PYTHON" tests/build_qwen_preprocess_fixture.py \
    --image "$IMAGE" \
    --out "$fixture" \
    --min-pixels "$min_pixels" \
    --max-pixels "$max_pixels"

  "$ZIG" build test \
    -Dtest-filter="qwen preprocessing parity" \
    -Dqwen-preprocess-fixture="$fixture" \
    --summary all
}

run_parity "$FIXTURE" "$MIN_PIXELS" "$MAX_PIXELS"

# The default fixture is close to a 1:1 resize. Force a substantial shrink as
# well so the reference check covers Pillow's scale-aware anti-aliasing path.
if ((DOWNSCALE_MIN_PIXELS > DOWNSCALE_MAX_PIXELS)); then
  DOWNSCALE_MIN_PIXELS="$DOWNSCALE_MAX_PIXELS"
fi
run_parity "$DOWNSCALE_FIXTURE" "$DOWNSCALE_MIN_PIXELS" "$DOWNSCALE_MAX_PIXELS"
