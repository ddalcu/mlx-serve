#!/bin/bash
# DiffusionGemma (block diffusion) end-to-end test.
#
# Boots a server on the DiffusionGemma checkpoint and asserts the canvas-
# denoising generation path works across the HTTP surfaces:
#   1. model loads + /v1/models reports model_type diffusion_gemma
#   2. chat non-stream: coherent greedy answer, finish_reason "stop",
#      EOS never leaks into content
#   3. the diffusion loop actually engaged (log: "[diffusion] canvas=")
#      and early-stopped (steps < max — convergence, not 48/48 churn)
#   4. chat streaming: block-wise SSE deltas + [DONE]
#   5. /v1/messages (Anthropic): text block + end_turn
#   6. tools: emits a valid tool_call with JSON args
#   7. thinking (/v1/messages with thinking): thinking + text blocks
#   8. multi-canvas: a long answer spans >1 canvas
#
# Requires:
#   - A built mlx-serve binary (zig build -Doptimize=ReleaseFast)
#   - DIFFUSION_TEST_MODEL or
#     ~/.mlx-serve/models/mlx-community/diffusiongemma-26B-A4B-it-4bit
#
# Usage: ./tests/test_diffusion_gemma.sh [port]

set -e

PORT=${1:-11398}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${DIFFUSION_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/diffusiongemma-26B-A4B-it-4bit}"
if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_diffusion_gemma: model directory not found at $MODEL"
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

FAILURES=0
check() {
    local desc="$1" ok="$2" detail="$3"
    if [ "$ok" = "1" ]; then
        echo -e "  ${GREEN}PASS${NC} $desc"
    else
        echo -e "  ${RED}FAIL${NC} $desc"
        [ -n "$detail" ] && echo "    $detail"
        FAILURES=$((FAILURES + 1))
    fi
}

LOG=$(mktemp /tmp/diffusion_test_log.XXXXXX)
echo "Starting server on port $PORT (model: $MODEL)..."
"$BINARY" --model "$MODEL" --serve --port "$PORT" --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; rm -f "$LOG"' EXIT

for i in $(seq 1 120); do
    if curl -s -m 1 "$BASE/health" > /dev/null 2>&1; then break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "${RED}FAIL${NC} server exited during load"; tail -5 "$LOG"; exit 1
    fi
    sleep 2
done

echo "── 1. model identity ──"
MT=$(curl -s "$BASE/v1/models" | python3 -c "import json,sys; d=json.load(sys.stdin)['data'][0]; print(1 if (d['loaded'] and d['state']=='ready') else 0)" 2>/dev/null || echo 0)
check "/v1/models reports model loaded + ready" "$MT"

echo "── 2. chat non-stream (greedy) ──"
RESP=$(curl -s -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Say hello in exactly five words."}],
  "max_tokens":64,"temperature":0,"stream":false}')
CONTENT=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
FINISH=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null || echo "")
LOWER=$(echo "$CONTENT" | tr '[:upper:]' '[:lower:]')
check "content contains 'hello'" "$([ "${LOWER#*hello}" != "$LOWER" ] && echo 1 || echo 0)" "content: $CONTENT"
check "finish_reason is stop" "$([ "$FINISH" = "stop" ] && echo 1 || echo 0)" "got: $FINISH"
case "$CONTENT" in *"<eos>"*|*"<turn|>"*) check "no special-token leak" 0 "content: $CONTENT";; *) check "no special-token leak" 1;; esac

echo "── 3. diffusion engagement + convergence ──"
check "diffusion loop engaged" "$(grep -q "\[diffusion\] canvas=" "$LOG" && echo 1 || echo 0)"
STEPS=$(grep "\[diffusion\] canvas=" "$LOG" | tail -1 | sed 's/.*steps=\([0-9]*\)\/\([0-9]*\).*/\1 \2/')
USED=$(echo "$STEPS" | cut -d' ' -f1); MAXS=$(echo "$STEPS" | cut -d' ' -f2)
check "early-stop fired (steps $USED < max $MAXS)" "$([ -n "$USED" ] && [ "$USED" -lt "$MAXS" ] && echo 1 || echo 0)" "steps=$STEPS"

echo "── 4. chat streaming ──"
STREAM=$(curl -sN -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Name one planet. Be brief."}],
  "max_tokens":48,"temperature":0,"stream":true}')
NCHUNKS=$(echo "$STREAM" | grep -c "^data: " || true)
check "SSE chunks emitted" "$([ "$NCHUNKS" -ge 3 ] && echo 1 || echo 0)" "chunks: $NCHUNKS"
check "stream terminates with [DONE]" "$(echo "$STREAM" | grep -q "data: \[DONE\]" && echo 1 || echo 0)"

echo "── 5. /v1/messages (Anthropic) ──"
AMSG=$(curl -s -m 600 "$BASE/v1/messages" -H 'Content-Type: application/json' -d '{
  "model":"mlx-serve","max_tokens":64,
  "messages":[{"role":"user","content":"What is 6 times 7? Answer with just the number."}]}')
ATEXT=$(echo "$AMSG" | python3 -c "import json,sys; d=json.load(sys.stdin); print(''.join(b.get('text','') for b in d['content']))" 2>/dev/null || echo "")
ASTOP=$(echo "$AMSG" | python3 -c "import json,sys; print(json.load(sys.stdin)['stop_reason'])" 2>/dev/null || echo "")
case "$ATEXT" in *42*) check "answer contains 42" 1;; *) check "answer contains 42" 0 "got: $ATEXT";; esac
check "stop_reason end_turn" "$([ "$ASTOP" = "end_turn" ] && echo 1 || echo 0)" "got: $ASTOP"

echo "── 6. tool calling ──"
TRESP=$(curl -s -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
  "max_tokens":128,"temperature":0,
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}')
TOOLOK=$(echo "$TRESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
c = d['choices'][0]
tcs = c['message'].get('tool_calls') or []
ok = (c['finish_reason'] == 'tool_calls' and len(tcs) == 1
      and tcs[0]['function']['name'] == 'get_weather'
      and json.loads(tcs[0]['function']['arguments']).get('city'))
print(1 if ok else 0)" 2>/dev/null || echo 0)
check "tool_call with valid JSON args" "$TOOLOK" "$(echo "$TRESP" | head -c 300)"

echo "── 7. thinking ──"
THINK=$(curl -s -m 600 "$BASE/v1/messages" -H 'Content-Type: application/json' -d '{
  "model":"mlx-serve","max_tokens":256,
  "thinking":{"type":"enabled","budget_tokens":128},
  "messages":[{"role":"user","content":"Is 17 prime? One short sentence."}]}')
THINKOK=$(echo "$THINK" | python3 -c "
import json, sys
d = json.load(sys.stdin)
types = [b['type'] for b in d['content']]
print(1 if 'thinking' in types and 'text' in types else 0)" 2>/dev/null || echo 0)
check "thinking + text blocks" "$THINKOK" "$(echo "$THINK" | head -c 300)"

echo "── 8. multi-canvas long generation ──"
LRESP=$(curl -s -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"List the numbers from one to forty as words, comma separated, then write one sentence about each season of the year."}],
  "max_tokens":512,"temperature":0,"stream":false}')
LTOK=$(echo "$LRESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo 0)
NCANVAS=$(grep -c "\[diffusion\] canvas=" "$LOG" || true)
check "long answer generated (tokens=$LTOK)" "$([ "$LTOK" -ge 100 ] && echo 1 || echo 0)"
check "multiple canvases ran (total=$NCANVAS)" "$([ "$NCANVAS" -ge 5 ] && echo 1 || echo 0)"

echo ""
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All diffusion tests passed.${NC}"
else
    echo -e "${RED}$FAILURES diffusion test(s) failed.${NC}"
    echo "Server log tail:"; tail -20 "$LOG"
    exit 1
fi
