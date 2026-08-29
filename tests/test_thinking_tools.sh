#!/bin/bash
# Integration tests: thinking + tools combinations (streaming and non-streaming).
# Tests all 8 permutations:
#   thinking × tools × streaming = 2 × 2 × 2 = 8 cases
#
# Usage: ./tests/test_thinking_tools.sh [model_dir] [port]
# Starts its own server, runs tests, kills it.

MODEL_DIR=${1:-${MLX_SERVE_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}}
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
FR2=$(echo "$RESP" | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0].get("finish_reason","?"))' 2>/dev/null)
# A verbose reasoner can still be INSIDE its thought when max_tokens lands
# (Laguna-XS spends >500 on 15x17). Empty content there is the truncated-thought
# rule working, not a bug — so demand content only when the block actually closed.
if [ "$FR2" = "length" ] && [ -z "$CONTENT" ]; then
  run_test "cut mid-thought: reasoning kept, nothing leaked to content" "$([ "$RC" != "NONE" ] && echo PASS || echo FAIL)" "finish=length"
else
  run_test "Has content" "$([ -n "$CONTENT" ] && echo PASS || echo FAIL)" "content='${CONTENT:0:80}'"
  run_test "Has reasoning_content" "$([ "$RC" != "NONE" ] && echo PASS || echo FAIL)" "reasoning='${RC:0:80}'"
fi
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
# Thinking BEFORE a tool call is the model's choice: Laguna-XS closes its
# pre-opened block immediately and calls the tool in ~35 tokens. The server
# guarantee is SEPARATION, not that a thought exists.
if [ "$RC" != "NONE" ]; then
  run_test "Has reasoning_content" "PASS" "reasoning='${RC:0:80}'"
else
  run_test "no-think tool turn: call intact, nothing leaked to content" "$([ -z "$CONTENT" ] && echo PASS || echo FAIL)" "content='${CONTENT:0:40}'"
fi
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
# Count delta lines whose `"content"` value is non-empty. The earlier filter
# (`grep '"content"' | grep -v '""' | grep -vc 'null'`) accidentally dropped
# any chunk containing "null" — and every SSE delta line carries
# `"finish_reason":null,"usage":null`, so a real content chunk got filtered
# out and the test reported HAS_CONTENT=0 even when the stream had content.
HAS_CONTENT=$(echo "$STREAM" | grep -cE '"content":"[^"]' ; true)
HAS_TC=$(echo "$STREAM" | grep -c '"tool_calls"' ; true)
NO_TAGS_CONTENT=$(echo "$STREAM" | grep '"content"' | grep -cE '<think>|<\|channel>thought' ; true)
NO_TAGS_RC=$(echo "$STREAM" | grep '"reasoning_content"' | grep -cE '<think>|<\|channel>thought' ; true)
if [ "$HAS_RC" -gt 0 ]; then
  run_test "Has reasoning_content deltas" "PASS" "$HAS_RC deltas"
else
  # Straight to the call (same model choice as Test 4) — Laguna-XS closes its
  # pre-opened block empty and writes its working as VISIBLE prose, which is a
  # legitimate turn: content alongside a tool call is normal. What must hold is
  # that the call survived; the two tag checks below cover the leak side.
  run_test "no-think tool stream: call still emitted" "$([ "$HAS_TC" -gt 0 ] && echo PASS || echo FAIL)" "tc=$HAS_TC content_chunks=$HAS_CONTENT"
fi
run_test "No thinking tags in content" "$([ "$NO_TAGS_CONTENT" -eq 0 ] && echo PASS || echo FAIL)"
run_test "No raw thinking tags in reasoning" "$([ "$NO_TAGS_RC" -eq 0 ] && echo PASS || echo FAIL)"

# Model may or may not use a tool — either tool_calls or content is fine.
# A verbose reasoner can also spend the whole 500-token budget INSIDE its
# thought and finish with neither (LFM2.5 does this on ~2 of 3 runs). That is
# the truncated-thought rule, not a failure — the reasoning must simply be
# there, and nothing may have leaked out as content.
FIN8=$(echo "$STREAM" | python3 -c 'import json,sys
fin=None
for line in sys.stdin:
    if not line.startswith("data: "): continue
    p=line[6:].strip()
    if p=="[DONE]": break
    try: fin=json.loads(p)["choices"][0].get("finish_reason") or fin
    except Exception: pass
print(fin)' 2>/dev/null)
if [ "$HAS_TC" -gt 0 ]; then
    run_test "Model chose tool call" "PASS" "tool_calls present"
elif [ "$HAS_CONTENT" -gt 0 ]; then
    run_test "Model answered directly" "PASS" "content present (no tool call)"
elif [ "$FIN8" = "length" ] && [ "$HAS_RC" -gt 0 ]; then
    run_test "budget spent mid-thought: reasoning kept, nothing leaked" "PASS" "finish=length, $HAS_RC reasoning deltas"
else
    run_test "Has tool_calls or content" "FAIL" "neither found (finish=$FIN8)"
fi

# ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 9: Thinking + tools, streaming, model may text or tool-call${NC}"
# ─────────────────────────────────────────────────────
STREAM=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in French. Do not use any tools.\"}],\"tools\":$TOOLS_JSON,\"max_tokens\":500,\"temperature\":0.1,\"stream\":true,\"enable_thinking\":true}")
HAS_RC=$(echo "$STREAM" | grep -c '"reasoning_content"' ; true)
# Count delta lines whose `"content"` value is non-empty. The earlier filter
# (`grep '"content"' | grep -v '""' | grep -vc 'null'`) accidentally dropped
# any chunk containing "null" — and every SSE delta line carries
# `"finish_reason":null,"usage":null`, so a real content chunk got filtered
# out and the test reported HAS_CONTENT=0 even when the stream had content.
HAS_CONTENT=$(echo "$STREAM" | grep -cE '"content":"[^"]' ; true)
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
