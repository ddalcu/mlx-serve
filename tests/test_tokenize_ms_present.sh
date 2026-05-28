#!/bin/bash
# test_tokenize_ms_present.sh — Iteration 1 of overnight perf push.
#
# The Phase 4 "instrumentation first" rule says: before we move chat-template
# render + tokenize off the request thread, we need to know how much time it
# actually takes. This test pins the existence of `timings.tokenize_ms` in
# every API surface (chat completions, anthropic messages, responses) and
# checks that it's a non-negative finite number.
#
# A pass-by-design test would only check presence; this also enforces a sane
# upper bound (< 1000 ms) so we catch the day someone wires it up to the
# wrong clock and writes a 1-second tokenize budget into a "5 short tokens"
# response.
#
# Env:
#   MODEL    Path to any MLX model dir. Default: Gemma 4 E4B 4-bit under
#            /Volumes/Sandisk_1TB/Models. Skipped if missing.
#   PORT     Server port. Default 19101.
#   BINARY   Path to mlx-serve. Default ./zig-out/bin/mlx-serve.

set -uo pipefail

PORT="${PORT:-19101}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${MODEL:-/Volumes/Sandisk_1TB/Models/mlx-community/gemma-4-e4b-it-4bit}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: model dir missing: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first ($BIN)"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }

pkill -9 -f "mlx-serve.*port $PORT" 2>/dev/null
sleep 1

LOG="$(mktemp)"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 -f "mlx-serve.*port $PORT" 2>/dev/null
    rm -f "$LOG"
}
trap cleanup EXIT INT TERM

"$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 4096 \
    --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 180); do
    curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; exit 1; }
    sleep 0.5
done

EC=0
RESULTS=()

# Pull `.timings.tokenize_ms` from a JSON response and verify it's a finite
# non-negative number under 1000 ms.
assert_tokenize_ms_in() {
    local label="$1" body="$2"
    local val
    val="$(echo "$body" | jq -r '.timings.tokenize_ms // empty')"
    if [ -z "$val" ]; then
        echo "  ❌ $label: timings.tokenize_ms missing"
        RESULTS+=("FAIL $label missing")
        EC=1
        return
    fi
    if ! python3 -c "import sys, math; v=float(sys.argv[1]); assert v >= 0 and v < 1000 and math.isfinite(v)" "$val" 2>/dev/null; then
        echo "  ❌ $label: tokenize_ms=$val (out of range)"
        RESULTS+=("FAIL $label range")
        EC=1
        return
    fi
    echo "  ✅ $label: tokenize_ms=${val}ms"
    RESULTS+=("PASS $label")
}

# --- /v1/chat/completions ---
echo "==> /v1/chat/completions (non-streaming)"
RESP="$(curl -sf --max-time 90 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":5,"temperature":0,"stream":false}')"
assert_tokenize_ms_in "chat-completions" "$RESP"

# --- /v1/messages (Anthropic) ---
echo "==> /v1/messages (non-streaming)"
RESP="$(curl -sf --max-time 90 -X POST "$BASE/v1/messages" \
    -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":5,"temperature":0,"stream":false}')"
assert_tokenize_ms_in "anthropic-messages" "$RESP"

# --- /v1/responses ---
echo "==> /v1/responses (non-streaming)"
RESP="$(curl -sf --max-time 90 -X POST "$BASE/v1/responses" \
    -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","input":"Say hi in one word.","max_output_tokens":16,"temperature":0,"stream":false}')"
assert_tokenize_ms_in "responses" "$RESP"

cleanup

echo
echo "=== summary ==="
for r in "${RESULTS[@]}"; do echo "  $r"; done
if [ "$EC" -ne 0 ]; then
    echo
    echo "FAIL: at least one API surface is missing timings.tokenize_ms or it's out of range."
fi
exit "$EC"
