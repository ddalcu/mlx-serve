#!/bin/bash
# Stopping one stream while another MTP stream decodes must not crash the server:
# the handler frees its sampling state (`think_bound`) once `complete` returns,
# and a tick that already holds the slot still reads it. Before `Slot.in_pass`
# this segfaulted in `thinkBoundTick` within two cancels.
#
# Usage: CANCEL_TEST_MODEL=<model with MTP + thinking> ./tests/test_cancel_mid_tick.sh [port]
set -u
MODEL="${CANCEL_TEST_MODEL:-/Volumes/G Drive SSD/models-dl/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit}"
PORT="${1:-11282}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
N="${CANCELS:-12}"
if [ ! -d "$MODEL" ]; then echo "skip: model not found ($MODEL)"; exit 0; fi

LOG=$(mktemp)
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --mtp --temp 0.7 --log-level info >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV $KEEP 2>/dev/null; rm -f "$LOG"' EXIT
for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done

body() { printf '{"model":"x","stream":true,"reasoning_effort":"low","max_tokens":1500,"messages":[{"role":"user","content":"%s"}]}' "$1"; }
curl -sN -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(body 'Write a very long detailed essay about the history of bridges.')" >/dev/null &
KEEP=$!
for i in $(seq 1 "$N"); do
    curl -sN -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(body "Explain in detail how a compiler works, part $i.")" >/dev/null &
    V=$!
    sleep 3
    kill $V 2>/dev/null
    wait $V 2>/dev/null
    sleep 0.3
    if ! kill -0 $SRV 2>/dev/null; then
        echo "FAIL: server died at cancel $i"
        tail -5 "$LOG"
        exit 1
    fi
done
echo "PASS: survived $N cancels beside a decoding MTP stream"
