#!/usr/bin/env bash
# Boot one model, fire the decode forward micro-bench at load, print the lines,
# shut down. Diagnostic helper for decode-perf work — NOT a test.
#
#   tests/fwd_ubench.sh <model-dir> [iters] [extra mlx-serve flags...]
#
# Env passthrough: any MLX_SERVE_* already exported reaches the server.
set -uo pipefail

MODEL="${1:?usage: fwd_ubench.sh <model-dir> [iters] [flags...]}"
ITERS="${2:-20}"
shift 2 2>/dev/null || shift 1
PORT="${PORT:-8099}"
BIN="${BIN:-./zig-out/bin/mlx-serve}"
LOG="${LOG:-/tmp/fwd-ubench-$PORT.log}"

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 1

MLX_SERVE_DECODE_FWD_UBENCH="$ITERS" "$BIN" --model "$MODEL" --serve --port "$PORT" \
  --log-level info "$@" >"$LOG" 2>&1 &
SRV=$!

for _ in $(seq 1 600); do
  grep -qE "\[fwd-ubench\]" "$LOG" && break
  kill -0 "$SRV" 2>/dev/null || break
  sleep 1
done
sleep 2

grep -E "\[fwd-ubench\]|\[ProjRung|\[moe\]|\[dtype\]|\[laguna|engaged|declined" "$LOG"

kill "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null
