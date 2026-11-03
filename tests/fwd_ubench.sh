#!/usr/bin/env bash
# fwd_ubench.sh — diagnostic, NOT a test: boot one model, fire diag decode-fwd-ubench at load, print ms/forward,
# the CPU graph-build vs GPU eval split, ops/forward and a terminal lm_head ablation, shut down.
#
#   tests/fwd_ubench.sh <model-dir> [iters] [extra mlx-serve flags...]
#
# The unit of every decode-perf A/B here: one run per boot per arm, ALTERNATING arms per boot (a block-of-N schedule
# reads drift as an effect). An exported MLX_SERVE_DIAG joins the server's: layer-cap=N (slope vs fixed cost),
# dispatch-probe=N (prices one dispatch with N exact-1.0 multiplies per MoE layer).

set -uo pipefail

MODEL="${1:?usage: fwd_ubench.sh <model-dir> [iters] [flags...]}"
ITERS="${2:-20}"
shift 2 2>/dev/null || shift 1
PORT="${PORT:-8099}"
BIN="${BIN:-./zig-out/bin/mlx-serve}"
LOG="${LOG:-/tmp/fwd-ubench-$PORT.log}"
[[ "$("$BIN" --version 2>/dev/null | head -1)" != *"(Debug)"* ]] || { echo "$BIN is a Debug build (2-4x slower) — build ReleaseFast first" >&2; exit 1; }

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 1

MLX_SERVE_DIAG="decode-fwd-ubench=$ITERS${MLX_SERVE_DIAG:+,$MLX_SERVE_DIAG}" "$BIN" --model "$MODEL" --serve --port "$PORT" \
  --log-level info "$@" >"$LOG" 2>&1 &
SRV=$!

for _ in $(seq 1 600); do
  grep -q "\[fwd-ubench\] done" "$LOG" && break
  kill -0 "$SRV" 2>/dev/null || break
  sleep 1
done
sleep 2

grep -E "\[fwd-ubench\]|\[ProjRung|\[moe\]|\[dtype\]|\[laguna|engaged|declined" "$LOG"

kill "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null
