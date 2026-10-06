#!/bin/bash
# Integration test: every request outcome is counted exactly once in /metrics.json.
#
#  1. A completed request            -> requests_success_total   +1, nothing else moves.
#  2. A stream the client drops
#     mid-decode                     -> requests_cancelled_total +1, nothing else moves.
#  3. A prompt over the context size
#     (refused before a slot exists) -> requests_rejected_total  +1, nothing else moves.
#  4. A 404 and a malformed body     -> no counter moves.
#
# Usage: ./tests/test_metrics_outcomes.sh [model_dir] [port]
#   Starts its own server. Default model: Gemma 4 E4B 8-bit.

set -u

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
PORT="${2:-11293}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_metrics_outcomes.log
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL (pass as first arg)"
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

"$BINARY" --model "$MODEL" --serve --port "$PORT" --metrics --no-pld --ctx-size 1024 --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "FAIL: server never became healthy on port $PORT"; exit 1; }

# "success cancelled failed rejected" from /metrics.json.
counters() {
    curl -s "$BASE/metrics.json" | python3 -c '
import json, sys
c = json.load(sys.stdin)["counters"]
print(*(c[k] for k in ("requests_success_total", "requests_cancelled_total",
                       "requests_failed_total", "requests_rejected_total")))'
}

# Wait until the counters stop moving (a dropped stream is counted on the inference thread, after the drop).
settled() {
    local prev cur
    prev=$(counters)
    for _ in $(seq 1 20); do
        sleep 1
        cur=$(counters)
        [ "$cur" = "$prev" ] && { echo "$cur"; return; }
        prev=$cur
    done
    echo "$prev"
}

# Wait for `field` (1-based index into counters) to reach `want`, up to 20 s.
wait_for() {
    local idx="$1" want="$2"
    for _ in $(seq 1 20); do
        [ "$(counters | cut -d' ' -f"$idx")" -ge "$want" ] && return 0
        sleep 1
    done
    return 1
}

BASELINE=$(settled)
echo "baseline (success cancelled failed rejected): $BASELINE"
read -r S0 C0 F0 R0 <<< "$BASELINE"

echo ""
echo "── 1. completed request ──"
curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":8}' >/dev/null
wait_for 1 $((S0 + 1))
read -r S1 C1 F1 R1 <<< "$(settled)"
check "success +1" "$([ "$S1" = "$((S0 + 1))" ] && echo 1 || echo 0)"
check "cancelled, failed, rejected unchanged" \
    "$([ "$C1" = "$C0" ] && [ "$F1" = "$F0" ] && [ "$R1" = "$R0" ] && echo 1 || echo 0)"

echo ""
echo "── 2. stream dropped mid-decode ──"
curl -sN --max-time 2 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a very long story about a lighthouse keeper."}],"max_tokens":600,"stream":true}' \
    >/dev/null 2>&1
wait_for 2 $((C1 + 1))
read -r S2 C2 F2 R2 <<< "$(settled)"
check "cancelled +1" "$([ "$C2" = "$((C1 + 1))" ] && echo 1 || echo 0)"
check "success, failed, rejected unchanged" \
    "$([ "$S2" = "$S1" ] && [ "$F2" = "$F1" ] && [ "$R2" = "$R1" ] && echo 1 || echo 0)"

echo ""
echo "── 3. refused at submit (prompt over --ctx-size) ──"
LONG=$(python3 -c 'print("lorem ipsum dolor sit amet " * 600)')
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$LONG\"}],\"max_tokens\":8}")
check "oversized prompt answers 400" "$([ "$CODE" = "400" ] && echo 1 || echo 0)"
read -r S3 C3 F3 R3 <<< "$(settled)"
check "rejected +1" "$([ "$R3" = "$((R2 + 1))" ] && echo 1 || echo 0)"
check "success, cancelled, failed unchanged" \
    "$([ "$S3" = "$S2" ] && [ "$C3" = "$C2" ] && [ "$F3" = "$F2" ] && echo 1 || echo 0)"

echo ""
echo "── 4. requests that are not completions ──"
curl -s -o /dev/null "$BASE/v1/no-such-endpoint"
curl -s -o /dev/null "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{not json'
read -r S4 C4 F4 R4 <<< "$(settled)"
check "404 and malformed body move no counter" \
    "$([ "$S4 $C4 $F4 $R4" = "$S3 $C3 $F3 $R3" ] && echo 1 || echo 0)"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
