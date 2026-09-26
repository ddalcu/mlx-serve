#!/bin/bash
# A qwen4exp (Qwen3.8-Flash-Next) GGUF that ds4 cannot load must be served by the
# embedded llama.cpp engine instead of taking the process down in ds4.
# Example fixture: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF IQ3_XXS (Q2_0 experts), shard 1.
#
# Gated on a fixture so CI without the model stays green:
#   QWEN4EXP_GGUF_MODEL=/path/to/model-00001-of-00002.gguf ./tests/test_qwen4exp_gguf_route.sh [port]
set -uo pipefail

MODEL="${QWEN4EXP_GGUF_MODEL:-}"
PORT="${1:-8124}"
BASE="http://127.0.0.1:$PORT"
BIN="./zig-out/bin/mlx-serve"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS=0; FAIL=0

if [ -z "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_qwen4exp_gguf_route: set QWEN4EXP_GGUF_MODEL=/path/to/model.gguf to run"
    exit 0
fi
if [ ! -f "$BIN" ]; then
    echo -e "${RED}ERROR${NC} $BIN not found — build with: zig build -Doptimize=ReleaseFast"
    exit 1
fi

ok()  { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
bad() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; [ -n "${2:-}" ] && echo "    $2"; }
assert_contains() { if grep -q "$2" <<< "$3"; then ok "$1"; else bad "$1" "missing '$2' in: $(echo "$3" | head -c 200)"; fi; }

LOG="$(mktemp)"
echo "→ starting mlx-serve on :$PORT with $MODEL"
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 16384 --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT

# A 75 GB checkpoint takes a while to map and warm.
for i in $(seq 1 300); do
    if curl -fs --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"'; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo -e "${RED}server died${NC}"; tail -20 "$LOG"; exit 1; fi
    sleep 1
done

assert_contains "routed to llama.cpp, not ds4" "\[gguf\] engine: llama (arch=qwen4exp" "$(cat "$LOG")"
assert_contains "llama engine loaded the model" "\[llama\] engine ready" "$(cat "$LOG")"

CHAT="$(curl -fs --max-time 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 17*23? Answer with just the number."}],"max_tokens":2048,"temperature":0}')"
assert_contains "chat completion answers" "391" "$CHAT"

TOOL="$(curl -fs --max-time 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Call get_weather for Paris."}],"max_tokens":2048,"temperature":0,
         "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}')"
assert_contains "tool call is emitted" '"name":"get_weather"' "$TOOL"
assert_contains "tool call carries the argument" 'Paris' "$TOOL"

echo "→ $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
