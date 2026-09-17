#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="${QWEN4_BF16_STREAM_MODEL:-$HOME/llm/models/Qwen/Qwen3.8-Flash-Next}"
OUT="${SSD_BENCH_OUT:-$REPO/ssd-fill-bench}"
SAMPLES="${SSD_BENCH_IOSTAT_SAMPLES:-180}"

if [ ! -d "$MODEL" ]; then
  echo "ssd_fill_bench: model dir not found: $MODEL" >&2
  exit 1
fi

mkdir -p "$OUT"
./.zig-toolchain/zig build test-build -Doptimize=ReleaseFast -Dtest-filter="expert io ssd"

date +%s.%N > "$OUT/iostat.start"
iostat -d 1 "$SAMPLES" > "$OUT/iostat.txt" &
IOSTAT_PID=$!
trap 'kill "$IOSTAT_PID" 2>/dev/null || true' EXIT

QWEN4_BF16_STREAM_MODEL="$MODEL" ./zig-out/tests/test 2>&1 | tee "$OUT/bench.txt"

kill "$IOSTAT_PID" 2>/dev/null || true
wait "$IOSTAT_PID" 2>/dev/null || true
echo "ssd_fill_bench: app-level rows in $OUT/bench.txt, physical MB/s in $OUT/iostat.txt (started $(cat "$OUT/iostat.start"))"
