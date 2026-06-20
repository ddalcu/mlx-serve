#!/usr/bin/env bash
# Qwen3-VL vision encoder parity vs the mlx-vlm reference.
#
# Builds a fixture (reference pixel_values + post-merger embeddings) with
# build_qwen_vision_fixture.py, then runs the QWEN_VISION_TEST_MODEL-gated Zig
# test which feeds the SAME pixel_values through QwenVision and asserts the
# embeddings match (mean-abs tight; a handful of bf16 reduction-order outliers
# tolerated — same discipline as the diffusion/MTP live parity tests).
#
# Usage:
#   tests/test_qwen_vision_parity.sh [model_dir] [image]
# Defaults to the local Qwen3.5-0.8B 4-bit checkpoint + house.jpeg. Skips
# cleanly (exit 0) when the model is absent.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
IMAGE="${2:-tests/fixtures/house.jpeg}"
FIX="${QWEN_VISION_FIXTURE:-/tmp/qwen_vision_fixture}"

if [ ! -f "$MODEL/config.json" ]; then
  echo "SKIP: model not found at $MODEL"
  exit 0
fi

echo "==> Building reference fixture ($IMAGE)"
python3 tests/build_qwen_vision_fixture.py --model "$MODEL" --image "$IMAGE" --out "$FIX"

# Pull the full patch grid (t, gh, gw) from the manifest.
read -r GH GW < <(python3 -c "import json;m=json.load(open('$FIX/manifest.json'));print(m['grid_thw'][1], m['grid_thw'][2])")
echo "==> grid_h=$GH grid_w=$GW"

echo "==> Running Zig parity test"
QWEN_VISION_TEST_MODEL="$MODEL" \
QWEN_VISION_FIXTURE="$FIX" \
QV_GH="$GH" QV_GW="$GW" \
  zig build test -Doptimize=ReleaseFast -Dtest-filter="qwen vision parity"

echo "PASS: qwen vision parity"
