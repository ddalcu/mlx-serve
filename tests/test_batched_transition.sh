#!/bin/bash
# Batched-decode transition consistency test.
#
# Regression for the legacy→batched decode-tick transition bug: a slot that
# starts generating alone (legacy single-slot pipelined decode) keeps a
# lookahead token ALREADY FORWARDED into its KV cache plus pending logits.
# When a second request arrives mid-generation, the slot joins a batched
# decode tick — which used to DROP that pending state and re-forward
# `next_token_id`, appending a duplicate position to the KV cache and
# re-emitting an already-emitted token. Symptom (llmprobe parity tests):
#
#   non-stream: Mercury,Venus,Earth
#       stream: MercuryMercury,Venus      <- dup first token, early stop
#
# This test:
#   1. captures a solo streaming baseline at temp=0
#   2. re-runs the same stream while injecting a concurrent request
#      mid-generation (legacy→batched transition), 3 rounds
#   3. fires two identical streams + one other request simultaneously
#      (fresh-from-prefill batch join), 2 rounds
# and asserts every captured stream equals the baseline byte-for-byte.
#
# Requires:
#   - A built mlx-serve binary (zig build -Doptimize=ReleaseFast)
#   - BATCH_TEST_MODEL or ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit
#
# Usage:
#   ./tests/test_batched_transition.sh [port]

set -e

PORT=${1:-8099}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${BATCH_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_batched_transition: model directory not found."
    echo "  Set BATCH_TEST_MODEL or place a checkpoint at ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit."
    exit 0
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

PROMPT="Count from 1 to 30, separated by single spaces. Output only the numbers."
OTHER_PROMPT="Name the largest ocean on Earth. Reply with just the name."

STREAM_BODY=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '$PROMPT'}],
    'max_tokens': 64,
    'temperature': 0.0,
    'stream': True,
}))
")
OTHER_BODY=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '$OTHER_PROMPT'}],
    'max_tokens': 16,
    'temperature': 0.0,
}))
")

sse_concat_content() {
    python3 -c '
import sys, json
out = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "):
        continue
    payload = line[6:].strip()
    if payload == "[DONE]" or not payload:
        continue
    try:
        ev = json.loads(payload)
    except Exception:
        continue
    for ch in ev.get("choices", []) or []:
        delta = ch.get("delta", {}) or {}
        text = delta.get("content")
        if isinstance(text, str):
            out.append(text)
sys.stdout.write("".join(out))
'
}

stream_request() {
    curl -s -N -m 120 -X POST -H "Content-Type: application/json" \
        -d "$STREAM_BODY" "$BASE/v1/chat/completions" | sse_concat_content
}

LOGFILE=$(mktemp)
"$BINARY" --model "$MODEL" --serve --port "$PORT" > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null || true; }
trap cleanup EXIT

up=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} server did not become healthy in 60s"
    tail -20 "$LOGFILE"
    exit 1
fi

echo "1) Solo streaming baseline..."
BASELINE=$(stream_request)
if [ -z "$BASELINE" ]; then
    echo -e "${RED}FAIL${NC} baseline stream produced no text"
    exit 1
fi
echo "   baseline: ${BASELINE:0:60}..."

FAILURES=0

check() {
    local label="$1" got="$2"
    if [ "$got" = "$BASELINE" ]; then
        echo -e "   ${GREEN}PASS${NC} $label"
    else
        echo -e "   ${RED}FAIL${NC} $label"
        echo "     expected: $BASELINE"
        echo "          got: $got"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "2) Mid-generation join (legacy→batched transition), 3 rounds..."
for round in 1 2 3; do
    TMP_A=$(mktemp)
    stream_request > "$TMP_A" &
    APID=$!
    sleep 0.4
    curl -s -m 60 -X POST -H "Content-Type: application/json" \
        -d "$OTHER_BODY" "$BASE/v1/chat/completions" > /dev/null
    wait $APID
    check "round $round (join mid-stream)" "$(cat "$TMP_A")"
    rm -f "$TMP_A"
done

echo "3) Simultaneous burst (fresh batch join), 2 rounds..."
for round in 1 2; do
    TMP_A=$(mktemp); TMP_B=$(mktemp)
    stream_request > "$TMP_A" &
    APID=$!
    stream_request > "$TMP_B" &
    BPID=$!
    curl -s -m 60 -X POST -H "Content-Type: application/json" \
        -d "$OTHER_BODY" "$BASE/v1/chat/completions" > /dev/null
    wait $APID $BPID
    check "burst round $round stream A" "$(cat "$TMP_A")"
    check "burst round $round stream B" "$(cat "$TMP_B")"
    rm -f "$TMP_A" "$TMP_B"
done

rm -f "$LOGFILE"
if [ "$FAILURES" -gt 0 ]; then
    echo -e "${RED}FAIL${NC} $FAILURES stream(s) diverged from solo baseline"
    exit 1
fi
echo -e "${GREEN}PASS${NC} all concurrent streams byte-identical to solo baseline"
