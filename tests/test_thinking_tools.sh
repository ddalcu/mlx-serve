#!/bin/bash
# Integration tests: thinking + tools combinations (streaming and non-streaming).
# Tests all 8 permutations:
#   thinking × tools × streaming = 2 × 2 × 2 = 8 cases
#
# Usage: ./tests/test_thinking_tools.sh [model_dir] [port]
# Starts its own server, runs tests, kills it.

MODEL_DIR=${1:-${MLX_SERVE_TEST_MODEL:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}}
PORT=${2:-8099}
BASE="http://127.0.0.1:$PORT"
BINARY="./zig-out/bin/mlx-serve"
PASS=0
FAIL=0
SKIP=0
TOTAL=0
LOG="/tmp/mlx-serve-test-thinking-tools.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
DIM='\033[2m'
NC='\033[0m'

run_test() {
    local name="$1" result="$2" detail="${3:-}"
    TOTAL=$((TOTAL + 1))
    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $name"
    elif [ "$result" = "SKIP" ]; then
        SKIP=$((SKIP + 1))
        echo -e "  ${YELLOW}SKIP${NC} $name — $detail"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $name"
        [ -n "$detail" ] && echo -e "    ${DIM}$detail${NC}"
    fi
}

echo "=== Thinking + Tools Integration Tests ==="
echo "Model: $MODEL_DIR"
echo "Port: $PORT"
echo ""

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

# Start server
echo "Starting server..."
$BINARY --model "$MODEL_DIR" --serve --port $PORT --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

for i in $(seq 1 30); do
    sleep 2
    curl -sf "$BASE/health" | grep -q ok && break
    if [ $i -eq 30 ]; then echo "FAIL: Server did not start"; exit 1; fi
done
echo -e "${GREEN}Server ready${NC}"
echo ""

TOOLS_JSON='[{"type":"function","function":{"name":"shell","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Command"}},"required":["command"]}}}]'

# ─────────────────────────────────────────────────────
echo -e "${YELLOW}Test 1: No thinking, no tools, non-streaming${NC}"
# ─────────────────────────────────────────────────────
RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":50,"temperature":0.1,"stream":false}')
CONTENT=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);m=d["choices"][0]["message"];print(m.get("content",""))' 2>/dev/null)
HAS_RC=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);m=d["choices"][0]["message"];print("yes" if m.get("reasoning_content") else "no")' 2>/dev/null)
run_test "Has content" "$([ -n "$CONTENT" ] && echo PASS || echo FAIL)" "content='$CONTENT'"
run_test "No reasoning_content" "$([ "$HAS_RC" = "no" ] && echo PASS || echo FAIL)"

# ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 2: Thinking enabled, no tools, non-streaming${NC}"
# ─────────────────────────────────────────────────────
RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 15 times 17?"}],"max_tokens":500,"temperature":0.1,"stream":false,"enable_thinking":true}')
CONTENT=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);m=d["choices"][0]["message"];print(m.get("content",""))' 2>/dev/null)
RC=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);m=d["choices"][0]["message"];rc=m.get("reasoning_content","");print(rc[:100] if rc else "NONE")' 2>/dev/null)
run_test "Has content" "$([ -n "$CONTENT" ] && echo PASS || echo FAIL)" "content='${CONTENT:0:80}'"
run_test "Has reasoning_content" "$([ "$RC" != "NONE" ] && echo PASS || echo FAIL)" "reasoning='${RC:0:80}'"
run_test "No thinking tags in content" "$(echo "$CONTENT" | grep -qE '<think>|<\|channel>' && echo FAIL || echo PASS)"

# ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 3: No thinking, tools enabled, non-streaming${NC}"
# ─────────────────────────────────────────────────────
RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Run the command: echo hello\"}],\"tools\":$TOOLS_JSON,\"max_tokens\":100,\"temperature\":0.1,\"stream\":false}")
FR=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0].get("finish_reason","?"))' 2>/dev/null)
TC_NAME=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);tcs=d["choices"][0]["message"].get("tool_calls",[]);print(tcs[0]["function"]["name"] if tcs else "NONE")' 2>/dev/null)
HAS_RC=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);m=d["choices"][0]["message"];print("yes" if m.get("reasoning_content") else "no")' 2>/dev/null)
run_test "Finish reason is tool_calls" "$([ "$FR" = "tool_calls" ] && echo PASS || echo FAIL)" "got '$FR'"
run_test "Tool call name is shell" "$([ "$TC_NAME" = "shell" ] && echo PASS || echo FAIL)" "got '$TC_NAME'"
run_test "No reasoning_content" "$([ "$HAS_RC" = "no" ] && echo PASS || echo FAIL)"

# ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 4: Thinking + tools, non-streaming${NC}"
# ─────────────────────────────────────────────────────
RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Run the command: echo hello\"}],\"tools\":$TOOLS_JSON,\"max_tokens\":500,\"temperature\":0.1,\"stream\":false,\"enable_thinking\":true}")
FR=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0].get("finish_reason","?"))' 2>/dev/null)
TC_NAME=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);tcs=d["choices"][0]["message"].get("tool_calls",[]);print(tcs[0]["function"]["name"] if tcs else "NONE")' 2>/dev/null)
RC=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);m=d["choices"][0]["message"];rc=m.get("reasoning_content","");print(rc[:100] if rc else "NONE")' 2>/dev/null)
CONTENT=$(echo "$RESP" | python3 -c 'import json,sys;d=json.load(sys.stdin);m=d["choices"][0]["message"];print(m.get("content") or "")' 2>/dev/null)
run_test "Tool call present" "$([ "$TC_NAME" = "shell" ] && echo PASS || echo FAIL)" "got '$TC_NAME'"
run_test "Has reasoning_content" "$([ "$RC" != "NONE" ] && echo PASS || echo FAIL)" "reasoning='${RC:0:80}'"
run_test "No thinking tags in content" "$(echo "$CONTENT" | grep -qE '<think>|<\|channel>' && echo FAIL || echo PASS)"

# ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 5: No thinking, no tools, streaming${NC}"
# ─────────────────────────────────────────────────────
STREAM=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":50,"temperature":0.1,"stream":true}')
HAS_CONTENT=$(echo "$STREAM" | grep -c '"content"' ; true)
HAS_RC=$(echo "$STREAM" | grep -c '"reasoning_content"' ; true)
HAS_DONE=$(echo "$STREAM" | grep -c '\[DONE\]' ; true)
run_test "Has content deltas" "$([ "$HAS_CONTENT" -gt 0 ] && echo PASS || echo FAIL)" "$HAS_CONTENT deltas"
run_test "No reasoning_content" "$([ "$HAS_RC" -eq 0 ] && echo PASS || echo FAIL)"
run_test "Has [DONE]" "$([ "$HAS_DONE" -gt 0 ] && echo PASS || echo FAIL)"

# ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 6: Thinking enabled, no tools, streaming${NC}"
# ─────────────────────────────────────────────────────
STREAM=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 15 times 17?"}],"max_tokens":500,"temperature":0.1,"stream":true,"enable_thinking":true}')
HAS_CONTENT=$(echo "$STREAM" | grep -c '"content"' ; true)
HAS_RC=$(echo "$STREAM" | grep -c '"reasoning_content"' ; true)
NO_TAGS=$(echo "$STREAM" | grep '"content"' | grep -cE '<think>|<\|channel>thought' ; true)
run_test "Has content deltas" "$([ "$HAS_CONTENT" -gt 0 ] && echo PASS || echo FAIL)"
run_test "Has reasoning_content deltas" "$([ "$HAS_RC" -gt 0 ] && echo PASS || echo FAIL)" "$HAS_RC deltas"
run_test "No thinking tags in content" "$([ "$NO_TAGS" -eq 0 ] && echo PASS || echo FAIL)"

# ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 7: No thinking, tools enabled, streaming${NC}"
# ─────────────────────────────────────────────────────
STREAM=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Run the command: echo hello\"}],\"tools\":$TOOLS_JSON,\"max_tokens\":100,\"temperature\":0.1,\"stream\":true}")
HAS_TC=$(echo "$STREAM" | grep -c '"tool_calls"' ; true)
HAS_RC=$(echo "$STREAM" | grep -c '"reasoning_content"' ; true)
FR=$(echo "$STREAM" | grep 'finish_reason' | grep -o '"tool_calls"\|"stop"' | head -1)
NO_TAGS=$(echo "$STREAM" | grep '"content"' | grep -cE '<think>|<\|channel>' ; true)
run_test "Has tool_calls delta" "$([ "$HAS_TC" -gt 0 ] && echo PASS || echo FAIL)"
run_test "No reasoning_content" "$([ "$HAS_RC" -eq 0 ] && echo PASS || echo FAIL)"
run_test "No thinking tags in content" "$([ "$NO_TAGS" -eq 0 ] && echo PASS || echo FAIL)"
run_test "Finish reason tool_calls" "$([ "$FR" = '"tool_calls"' ] && echo PASS || echo FAIL)" "got $FR"

# ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 8: Thinking + tools, streaming (THE FIX)${NC}"
# ─────────────────────────────────────────────────────
STREAM=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 15 times 17? Think step by step then use shell to verify with: echo \$((15*17))\"}],\"tools\":$TOOLS_JSON,\"max_tokens\":500,\"temperature\":0.1,\"stream\":true,\"enable_thinking\":true}")
HAS_RC=$(echo "$STREAM" | grep -c '"reasoning_content"' ; true)
HAS_CONTENT=$(echo "$STREAM" | grep '"content"' | grep -v '""' | grep -vc 'null' ; true)
HAS_TC=$(echo "$STREAM" | grep -c '"tool_calls"' ; true)
NO_TAGS_CONTENT=$(echo "$STREAM" | grep '"content"' | grep -cE '<think>|<\|channel>thought' ; true)
NO_TAGS_RC=$(echo "$STREAM" | grep '"reasoning_content"' | grep -cE '<think>|<\|channel>thought' ; true)
run_test "Has reasoning_content deltas" "$([ "$HAS_RC" -gt 0 ] && echo PASS || echo FAIL)" "$HAS_RC deltas"
run_test "No thinking tags in content" "$([ "$NO_TAGS_CONTENT" -eq 0 ] && echo PASS || echo FAIL)"
run_test "No raw thinking tags in reasoning" "$([ "$NO_TAGS_RC" -eq 0 ] && echo PASS || echo FAIL)"

# Model may or may not use a tool — either tool_calls or content is fine
if [ "$HAS_TC" -gt 0 ]; then
    run_test "Model chose tool call" "PASS" "tool_calls present"
elif [ "$HAS_CONTENT" -gt 0 ]; then
    run_test "Model answered directly" "PASS" "content present (no tool call)"
else
    run_test "Has tool_calls or content" "FAIL" "neither found"
fi

# ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 9: Thinking + tools, streaming, model may text or tool-call${NC}"
# ─────────────────────────────────────────────────────
STREAM=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in French. Do not use any tools.\"}],\"tools\":$TOOLS_JSON,\"max_tokens\":500,\"temperature\":0.1,\"stream\":true,\"enable_thinking\":true}")
HAS_RC=$(echo "$STREAM" | grep -c '"reasoning_content"' ; true)
HAS_CONTENT=$(echo "$STREAM" | grep '"content"' | grep -v '""' | grep -vc 'null' ; true)
HAS_TC=$(echo "$STREAM" | grep -c '"tool_calls"' ; true)
NO_TAGS=$(echo "$STREAM" | grep '"content"' | grep -cE '<think>|<\|channel>' ; true)
HAS_ANY=$(echo "$STREAM" | grep -c 'data:' ; true)
run_test "Stream has data events" "$([ "$HAS_ANY" -gt 0 ] && echo PASS || echo FAIL)" "events=$HAS_ANY rc=$HAS_RC content=$HAS_CONTENT tc=$HAS_TC"
run_test "No thinking tags in content" "$([ "$NO_TAGS" -eq 0 ] && echo PASS || echo FAIL)"

# ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo -e "  ${GREEN}Passed: $PASS${NC}  ${RED}Failed: $FAIL${NC}  ${YELLOW}Skipped: $SKIP${NC}  Total: $TOTAL"
echo "═══════════════════════════════════════════════"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
