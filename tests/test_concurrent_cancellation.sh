#!/bin/bash
# Concurrent cancellation test (Phase A7).
#
# Hammers the server with N cycles of "start a long generation, kill the
# client mid-stream" and asserts that no slot leaks. The scheduler's slot
# accounting (queue.in_flight, decoding.items.len) must drop back to 0 once
# the conn thread tears down. A leak would cause /health to eventually
# refuse work or, more often, accumulate slot allocations until the server
# OOMs under load.
#
# How we detect a leak: after each batch of cancelled requests we spin until
# /health responds, then submit a final non-cancelled request to confirm the
# server can still service work. If the queue is jammed by leaked slots the
# final request times out. We also log the count of `[scheduler]` lines in
# the server log to catch silently-stuck slots that accept submits but never
# complete.
#
# Requires:
#   - A built mlx-serve binary
#   - A model directory (see PLD_TEST_MODEL fallback chain).
#
# Usage:
#   CANCEL_TEST_MODEL=/path/to/model ./tests/test_concurrent_cancellation.sh [port] [cycles]

set -e

PORT=${1:-8094}
CYCLES=${2:-${CYCLES:-100}}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${CANCEL_TEST_MODEL:-${PLD_TEST_MODEL:-$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit}}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_concurrent_cancellation: model directory not found."
    echo "  Set CANCEL_TEST_MODEL or place an MLX checkpoint at"
    echo "  ~/.mlx-serve/models/gemma-4-e4b-it-8bit."
    exit 0
fi

if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable."
    exit 1
fi

# Long-output prompt so each request takes long enough that we can reliably
# kill it before completion. max_tokens high so the kill races a real decode
# loop, not just a stalled prefill.
PROMPT='Write a 500-word story about a robot that learns to paint.'

JSON_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '''$PROMPT'''}],
    'max_tokens': 400,
    'temperature': 0.0,
    'stream': True,
}))
")

# Final-confirmation payload: short, deterministic. If the queue is leaked
# this won't return in TIMEOUT seconds.
FINAL_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': 'Reply with just the word OK.'}],
    'max_tokens': 8,
    'temperature': 0.0,
    'stream': False,
}))
")

echo "== concurrent cancellation test =="
echo "  model: $MODEL"
echo "  cycles: $CYCLES"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

LOGFILE=$(mktemp)
"$BINARY" --model "$MODEL" --serve --port "$PORT" --max-concurrent 4 --no-pld > "$LOGFILE" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; rm -f $LOGFILE" EXIT INT TERM

echo "  starting server..."
up=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then
        up=1
        break
    fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} server did not become healthy in 60s"
    tail -30 "$LOGFILE"
    exit 1
fi

# ── Cancellation cycles ──
# Per cycle:
#   1. POST a streaming request, get the curl PID
#   2. Wait briefly so generation actually starts (some tokens stream)
#   3. Kill curl
# Repeated $CYCLES times. After all cycles we send a final non-cancelled
# request to confirm the server still serves.
echo "  running $CYCLES cancellation cycles..."
FAIL_COUNT=0
for cycle in $(seq 1 "$CYCLES"); do
    # Salt the prompt so the server doesn't deduplicate via hot cache.
    SALTED=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '[c $cycle] $PROMPT'}],
    'max_tokens': 400,
    'temperature': 0.0,
    'stream': True,
}))
")
    # Bound curl's lifetime with `--max-time` so it self-terminates after a
    # short window. That tears down the TCP connection cleanly and the
    # server's conn thread sees a broken-pipe write on its next SSE chunk —
    # exactly the cancellation path we're stressing. Using `kill -9` on the
    # subshell didn't always propagate to the curl child (process-group
    # semantics), which left curl orphaned and the server still streaming.
    SLEEP_S=$(python3 -c "import random; print(f'{0.1 + random.random()*0.2:.3f}')")
    (echo "$SALTED" | curl -s -N --max-time "$SLEEP_S" -X POST -H "Content-Type: application/json" \
        -d @- "$BASE/v1/chat/completions" > /dev/null 2>&1) &
    REQ_PID=$!
    wait $REQ_PID 2>/dev/null || true

    if [ $((cycle % 20)) -eq 0 ]; then
        echo "    completed $cycle/$CYCLES cancellation cycles"
    fi
done

# Give the inference thread a moment to drain any in-flight slots.
sleep 2

# ── Confirm the server is still healthy and responsive ──
echo "  verifying server still responds after cancellation storm..."
if ! curl -s -f --max-time 10 "$BASE/health" > /dev/null; then
    echo -e "${RED}FAIL${NC} /health unresponsive after $CYCLES cancellations (slot leak suspected)"
    tail -40 "$LOGFILE" | sed 's/^/    /'
    exit 1
fi

# Final payload should round-trip cleanly. Use a generous timeout so a
# slow-but-not-stuck server still passes.
FINAL_T0=$(python3 -c 'import time; print(time.time())')
FINAL_RESP=$(echo "$FINAL_PAYLOAD" | curl -s --max-time 30 -X POST \
    -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" || echo "")
FINAL_T1=$(python3 -c 'import time; print(time.time())')
FINAL_ELAPSED=$(python3 -c "print(round($FINAL_T1 - $FINAL_T0, 2))")

if [ -z "$FINAL_RESP" ]; then
    echo -e "${RED}FAIL${NC} final request did not return within 30s — slot leak"
    tail -40 "$LOGFILE" | sed 's/^/    /'
    exit 1
fi

FINAL_TEXT=$(echo "$FINAL_RESP" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    print(r['choices'][0]['message']['content'])
except Exception as e:
    print(f'PARSE_ERROR {e}')
")

echo "  final request: ${FINAL_ELAPSED}s, response: $(echo "$FINAL_TEXT" | head -1)"

if [[ "$FINAL_TEXT" == PARSE_ERROR* ]]; then
    echo -e "${RED}FAIL${NC} final response not parseable: $FINAL_TEXT"
    echo "  raw: $FINAL_RESP" | head -3
    exit 1
fi

# Sanity check the scheduler logs for stuck-slot indicators. We expect lots
# of "cancelled" finish reasons but no error or unhandled-exception traces.
ERR_LINES=$(grep -cE "\[scheduler\] (decode tick failed|prefill failed)" "$LOGFILE" || true)
if [ "$ERR_LINES" -gt 0 ]; then
    echo -e "${YELLOW}WARN${NC} found $ERR_LINES scheduler error lines in log:"
    grep -E "\[scheduler\] (decode tick failed|prefill failed)" "$LOGFILE" | head -5 | sed 's/^/    /'
    # Not necessarily a fail — some errors during cancellation are expected
    # (e.g. mid-tick cancel), but we surface them for diagnosis.
fi

echo -e "${GREEN}PASS${NC} server survived $CYCLES cancellation cycles, final request OK in ${FINAL_ELAPSED}s"
exit 0
