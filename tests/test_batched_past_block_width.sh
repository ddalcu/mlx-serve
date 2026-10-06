#!/bin/bash
# A batched group as wide as the drafter's block survives beside a speculating slot.
#
# The bar: 20 simultaneous streams on a drafter-bound GDN pack all finish with content and the
# server logs no aborted batched tick. (A kernel config cached by row count once handed a
# 16-slot tick a 16-wide verify's output layout; see docs/gotchas/engine-mlx.md.)
#
# Usage: [BLOCK_WIDTH_MODEL=<model-dir>] ./tests/test_batched_past_block_width.sh [port]
set -u
source "$(dirname "$0")/_lib_models.sh"
MODEL="${BLOCK_WIDTH_MODEL:-$(find_fitting_model ddalcu/Qwen3.8-27B-MLX-Serve-4bit)}"
PORT="${1:-11433}"
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
STREAMS=20
[ -d "$MODEL" ] || { echo "SKIP test_batched_past_block_width: no fitting 27B pack"; exit 0; }
[ -d "$MODEL/drafter" ] || { echo "SKIP test_batched_past_block_width: $MODEL has no drafter/"; exit 0; }
LOG=$(mktemp); OUT=$(mktemp -d)
"$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --max-concurrent 32 --log-level info > "$LOG" 2>&1 &
SRV=$!
cleanup() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; rm -rf "$LOG" "$OUT"; }
trap cleanup EXIT
for _ in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL server did not become healthy"; tail -20 "$LOG"; exit 1; }
EXCERPT=$(head -c 5000 src/pld_index.zig | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')
for i in $(seq 1 $STREAMS); do
    printf '{"model":"m","stream":true,"max_tokens":128,"temperature":0.7,"messages":[{"role":"user","content":"Reviewer %d of %d. Explain this Zig excerpt and list three improvements:\\n\\n%s"}]}' "$i" "$STREAMS" "$(echo "$EXCERPT" | sed 's/^"//; s/"$//')" > "$OUT/req$i.json"
    curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$OUT/req$i.json" > "$OUT/resp$i.sse" &
done
wait
FAIL=0
for i in $(seq 1 $STREAMS); do
    n=$(/usr/bin/grep -c '"content":"' "$OUT/resp$i.sse" 2>/dev/null || echo 0)
    [ "$n" -ge 8 ] || { echo "FAIL stream $i delivered $n content chunks"; FAIL=1; }
done
aborted=$(/usr/bin/grep -c "decode aborted\|streaming error\|verify aborted" "$LOG")
[ "$aborted" = "0" ] || { echo "FAIL server logged $aborted aborted/errored streams"; /usr/bin/grep -m3 "aborted\|concatenate" "$LOG"; FAIL=1; }
[ "$FAIL" = "0" ] && echo "PASS test_batched_past_block_width: $STREAMS streams complete, no aborted batched tick" || exit 1
