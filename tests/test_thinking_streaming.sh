#!/bin/bash
# Integration tests for thinking-tag streaming behavior.
# Verifies fixes for:
#   - Templates that pre-inject the opener (Qwen 3.5/3.6) — model output starts inside the think block
#   - Templates that emit the opener literally (Gemma 4) — `<|channel>thought` prefix must be stripped
#   - Streaming reasoning_content vs content split in real time
#   - Streaming reasoning + tools (most complex path)
#
# Usage: ./tests/test_thinking_streaming.sh [model_dir] [port]
# Starts its own server, runs tests, kills it.

MODEL_DIR=${1:-${MLX_SERVE_TEST_MODEL:-$HOME/.mlx-serve/models/unsloth/Qwen3.6-27B-UD-MLX-4bit}}
PORT=${2:-8099}
BASE="http://127.0.0.1:$PORT"
BINARY="./zig-out/bin/mlx-serve"
PASS=0
FAIL=0
TOTAL=0
LOG="/tmp/mlx-serve-thinking-test.log"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
DIM='\033[2m'
NC='\033[0m'

check() {
    local name="$1" condition="$2" detail="${3:-}"
    TOTAL=$((TOTAL + 1))
    if [ "$condition" = "true" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $name"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $name"
        [ -n "$detail" ] && echo -e "    ${DIM}$detail${NC}"
    fi
}

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

if [ ! -f "$BINARY" ]; then
    echo "Building..."
    zig build -Doptimize=ReleaseFast 2>&1 || { echo "Build failed"; exit 1; }
fi

echo "=== Thinking Streaming Integration Tests ==="
echo "Model: $MODEL_DIR"
echo "Port:  $PORT"
echo ""

"$BINARY" --model "$MODEL_DIR" --serve --port $PORT --log-level warn --ctx-size 8192 2>"$LOG" &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 60); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then break; fi
    if [ $i -eq 60 ]; then echo "FAIL: server did not start"; exit 1; fi
    sleep 1
done
echo -e "${GREEN}Server ready${NC}"
echo ""

# Detect whether the model actually emits a `<think>` close tag (some quants
# under-think and finish without one). We probe with a simple prompt.
analyze_stream() {
    local resp_file="$1" prefix="$2"
    python3 -c "
import json, sys
prefix = sys.argv[1]
path = sys.argv[2]
content_chunks, reasoning_chunks, tool_calls = [], [], []
finish = None
with open(path, 'rb') as f:
    raw = f.read().decode('utf-8', errors='replace')
for line in raw.splitlines():
    line = line.strip()
    if not line.startswith('data: '): continue
    payload = line[6:]
    if payload == '[DONE]': break
    try: d = json.loads(payload)
    except Exception: continue
    choices = d.get('choices') or []
    if not choices: continue
    delta = choices[0].get('delta', {})
    fr = choices[0].get('finish_reason')
    if fr: finish = fr
    if delta.get('content'): content_chunks.append(delta['content'])
    if delta.get('reasoning_content'): reasoning_chunks.append(delta['reasoning_content'])
    if delta.get('tool_calls'): tool_calls.extend(delta['tool_calls'])
content_full = ''.join(content_chunks)
reasoning_full = ''.join(reasoning_chunks)
tags = ('<think', '</think', '<|channel>thought', '<|channel>', '<channel|>')
print(f'{prefix}_finish={finish}')
print(f'{prefix}_reasoning_len={len(reasoning_full)}')
print(f'{prefix}_reasoning_chunks={len(reasoning_chunks)}')
print(f'{prefix}_content_len={len(content_full)}')
print(f'{prefix}_content_chunks={len(content_chunks)}')
print(f'{prefix}_reasoning_has_tag={any(t in reasoning_full for t in tags)}')
print(f'{prefix}_content_has_tag={any(t in content_full for t in tags + (chr(60)+\"tool_call\",))}')
print(f'{prefix}_tool_calls={len(tool_calls)}')
" "$prefix" "$resp_file"
}

# ─────────────────────────────────────────────────────────────────────
# Test 1: non-streaming + thinking, simple math
# ─────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}Test 1: non-streaming + thinking${NC}"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 17 * 23? Briefly."}],"max_tokens":1500,"temperature":0.3,"stream":false,"enable_thinking":true}')
CONTENT=$(echo "$RESULT" | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print((m.get('content') or '')[:400])")
REASONING=$(echo "$RESULT" | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print(len(m.get('reasoning_content') or ''))")
CONTENT_HAS_TAG=$(echo "$CONTENT" | python3 -c "import sys; t=sys.stdin.read(); print('true' if any(x in t for x in ('<think','</think','<|channel','<channel|')) else 'false')")
check "reasoning_content populated (>50 chars)" "$([ ${REASONING:-0} -gt 50 ] && echo true || echo false)" "got $REASONING"
check "content has no thinking tags" "$([ "$CONTENT_HAS_TAG" = "false" ] && echo true || echo false)" "$CONTENT"

# ─────────────────────────────────────────────────────────────────────
# Test 2: streaming + thinking, no tools
# ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 2: streaming + thinking, no tools${NC}"
curl -sN "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 17 * 23? Just the number after thinking."}],"max_tokens":1500,"temperature":0.3,"stream":true,"enable_thinking":true}' \
  > /tmp/think_stream_2.txt 2>&1
eval "$(analyze_stream /tmp/think_stream_2.txt t2)"
# Whether the model thinks at all is a MODEL choice on families whose template
# has no enable_thinking hook: Gemma 4 renders a bare `<|turn>model\n` and
# decides for itself, and on this prompt e4b answers "391" directly. Templates
# that pre-inject the opener (Qwen 3.5/3.6, LFM2.5) always produce a block.
# So assert the invariant that holds either way — a turn is EITHER a streamed
# think block, OR a direct answer as content — and never the broken third
# state this test used to pass through: the whole answer filed as
# reasoning_content with empty content (live Gemma 2026-08-04, streaming only;
# non-streaming had it right, which is what made the two paths disagree).
if [ ${t2_reasoning_len:-0} -gt 0 ]; then
  check "streaming reasoning_content populated (>50)" "$([ ${t2_reasoning_len:-0} -gt 50 ] && echo true || echo false)" "got $t2_reasoning_len"
  check "streaming reasoning chunks > 1 (live streaming)" "$([ ${t2_reasoning_chunks:-0} -gt 1 ] && echo true || echo false)" "chunks=$t2_reasoning_chunks"
else
  check "no-think turn: answer arrives as CONTENT, not reasoning" "$([ ${t2_content_len:-0} -gt 0 ] && echo true || echo false)" "content_len=$t2_content_len"
  check "no-think turn: reasoning_content stays empty" "$([ ${t2_reasoning_chunks:-0} -eq 0 ] && echo true || echo false)" "chunks=$t2_reasoning_chunks"
fi
check "streaming reasoning_content has no leaked tags" "$([ "$t2_reasoning_has_tag" = "False" ] && echo true || echo false)" ""
check "streaming content has no leaked tags" "$([ "$t2_content_has_tag" = "False" ] && echo true || echo false)" ""

# ─────────────────────────────────────────────────────────────────────
# Test 3: streaming + thinking + tools
# ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 3: streaming + thinking + tools${NC}"
curl -sN "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What time is it? Use the shell tool."}],"max_tokens":800,"temperature":0.3,"stream":true,"enable_thinking":true,"tools":[{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]}' \
  > /tmp/think_stream_3.txt 2>&1
eval "$(analyze_stream /tmp/think_stream_3.txt t3)"
check "thinking+tools: emitted at least one tool_call" "$([ ${t3_tool_calls:-0} -ge 1 ] && echo true || echo false)" "tool_calls=$t3_tool_calls"
check "thinking+tools: content has no tag/tool-call leak" "$([ "$t3_content_has_tag" = "False" ] && echo true || echo false)" ""
check "thinking+tools: reasoning_content has no tag leak" "$([ "$t3_reasoning_has_tag" = "False" ] && echo true || echo false)" ""

# ─────────────────────────────────────────────────────────────────────
# Test 4: non-streaming + thinking + tool ROUND-TRIP (turn 2 after tool result)
# Reproduces the Qwen 3.6 bug where the second turn returned empty content.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 4: non-streaming + thinking + tool round-trip${NC}"
RT_BODY='{
  "model":"mlx-serve",
  "messages":[
    {"role":"user","content":"What time is it? Use the shell tool."},
    {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"date\"}"}}]},
    {"role":"tool","tool_call_id":"call_1","content":"Fri Apr 17 08:15:00 PDT 2026"}
  ],
  "max_tokens":500,
  "temperature":0.3,
  "stream":false,
  "enable_thinking":true,
  "tools":[{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
}'
RT_RESULT=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$RT_BODY")
RT_CONTENT=$(echo "$RT_RESULT" | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print(len(m.get('content') or ''))")
RT_REASONING=$(echo "$RT_RESULT" | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print(len(m.get('reasoning_content') or ''))")
check "round-trip non-stream content populated (>5)" "$([ ${RT_CONTENT:-0} -gt 5 ] && echo true || echo false)" "got $RT_CONTENT"
# Reasoning is optional — model may legitimately skip thinking on simple follow-ups.

# ─────────────────────────────────────────────────────────────────────
# Test 5: STREAMING + thinking + tool ROUND-TRIP (the bug the user hit)
# ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Test 5: streaming + thinking + tool round-trip${NC}"
RT_STREAM_BODY=$(echo "$RT_BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); d['stream']=True; print(json.dumps(d))")
curl -sN "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$RT_STREAM_BODY" > /tmp/think_stream_5.txt 2>&1
eval "$(analyze_stream /tmp/think_stream_5.txt t5)"
check "round-trip stream content populated (>5)" "$([ ${t5_content_len:-0} -gt 5 ] && echo true || echo false)" "got $t5_content_len"
# Reasoning is optional — model may legitimately skip thinking on simple follow-ups.
check "round-trip stream content has no tag leak" "$([ "$t5_content_has_tag" = "False" ] && echo true || echo false)" ""
check "round-trip stream reasoning has no tag leak" "$([ "$t5_reasoning_has_tag" = "False" ] && echo true || echo false)" ""

echo ""
echo "═══════════════════════════════════════════════"
echo -e "  ${GREEN}Passed: $PASS${NC}  ${RED}Failed: $FAIL${NC}  Total: $TOTAL"
echo "═══════════════════════════════════════════════"

[ $FAIL -eq 0 ] && exit 0 || exit 1
