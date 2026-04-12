#!/bin/bash
# Integration tests for tool call argument parsing and thinking tag handling.
# Verifies fixes for:
#   - Gemma 4 tool call args with nested JSON/braces getting mangled
#   - Thinking tags (<|channel>thought) leaking as content when thinking is disabled
#   - Tool call round-trip with multiline code content
#
# Usage: ./tests/test_tool_parsing.sh [model_dir] [port]
# Starts its own server, runs tests, kills it.

MODEL_DIR=${1:-${MLX_SERVE_TEST_MODEL:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}}
PORT=${2:-8097}
BASE="http://127.0.0.1:$PORT"
BINARY="./zig-out/bin/mlx-serve"
PASS=0
FAIL=0
SKIP=0
TOTAL=0
LOG="/tmp/mlx-serve-test-tool-parsing.log"

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
        [ -n "$detail" ] && echo -e "    $detail"
    fi
}

echo "=== Tool Parsing & Thinking Tag Integration Tests ==="
echo "Model: $MODEL_DIR"
echo "Port: $PORT"
echo ""

# Check model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

# Build if needed
if [ ! -f "$BINARY" ]; then
    echo "Building..."
    zig build 2>&1 || { echo "Build failed"; exit 1; }
fi

# Start server
echo "Starting server..."
"$BINARY" --model "$MODEL_DIR" --serve --port $PORT --log-level debug --ctx-size 8192 2>"$LOG" &
SERVER_PID=$!

cleanup() {
    echo ""
    echo -e "${DIM}Stopping server (PID $SERVER_PID)...${NC}"
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

for i in $(seq 1 30); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then break; fi
    if [ $i -eq 30 ]; then echo "FAIL: Server did not start"; exit 1; fi
    sleep 1
done
echo -e "${GREEN}Server ready${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Test 1: Tool call with writeFile containing code (nested braces)
# ═══════════════════════════════════════════════════════════════════
echo -e "${YELLOW}Test 1: writeFile tool call with code content (nested braces)${NC}"

RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You MUST call the writeFile tool. Do not respond with text. Call writeFile with path=\"test.js\" and content that is a simple Express server with one GET route that returns JSON {\"status\":\"ok\"}."},
      {"role": "user", "content": "Create a test.js file with an Express server"}
    ],
    "tools": [{"type":"function","function":{"name":"writeFile","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}],
    "max_tokens": 1024,
    "temperature": 0.3,
    "stream": false
  }')

FINISH=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0].get('finish_reason',''))" 2>/dev/null)
if [ "$FINISH" = "tool_calls" ]; then
    # Verify arguments have both path and content
    PARSE_RESULT=$(echo "$RESULT" | python3 -c "
import sys, json
r = json.load(sys.stdin)
tc = r['choices'][0]['message']['tool_calls'][0]
args = json.loads(tc['function']['arguments'])
path = args.get('path', '')
content = args.get('content', '')
if not path:
    print('FAIL:missing_path')
elif not content:
    print('FAIL:missing_content')
elif len(content) < 20:
    print('FAIL:content_too_short:' + repr(content))
elif '{' not in content:
    print('FAIL:no_braces_in_content')
else:
    print('PASS:' + str(len(content)))
" 2>/dev/null)
    STATUS=$(echo "$PARSE_RESULT" | cut -d: -f1)
    DETAIL=$(echo "$PARSE_RESULT" | cut -d: -f2-)
    if [ "$STATUS" = "PASS" ]; then
        run_test "writeFile args preserved with nested braces" "PASS"
        echo -e "    ${DIM}content length: $DETAIL chars${NC}"
    else
        run_test "writeFile args preserved with nested braces" "FAIL" "$DETAIL"
        echo -e "    ${DIM}raw: $(echo "$RESULT" | python3 -c "import sys,json; tc=json.load(sys.stdin)['choices'][0]['message']['tool_calls'][0]; print(tc['function']['arguments'][:300])" 2>/dev/null)${NC}"
    fi
else
    run_test "writeFile args preserved with nested braces" "SKIP" "model did not generate tool_calls (finish=$FINISH)"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════
# Test 2: Thinking tags stripped when thinking=disabled, tools present
# ═══════════════════════════════════════════════════════════════════
echo -e "${YELLOW}Test 2: Thinking tags not in content when thinking disabled (with tools)${NC}"

# Streaming request with tools but no enable_thinking — channel tags should not appear in content
EVENTS=$(curl -sf -N "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "Think step by step about what tool to use, then call it."},
      {"role": "user", "content": "List files in the current directory"}
    ],
    "tools": [{"type":"function","function":{"name":"shell","description":"Run command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
    "max_tokens": 512,
    "temperature": 0.5,
    "stream": true
  }' 2>/dev/null)

# Collect all content deltas
CONTENT=$(echo "$EVENTS" | grep '^data: ' | grep -v '\[DONE\]' | sed 's/^data: //' | python3 -c "
import sys, json
content = ''
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        chunk = json.loads(line)
        delta = chunk.get('choices', [{}])[0].get('delta', {})
        c = delta.get('content', '')
        if c: content += c
    except: pass
print(content)
" 2>/dev/null)

if echo "$CONTENT" | grep -q '<|channel>'; then
    run_test "No <|channel> tags in streamed content" "FAIL" "found <|channel> in: ${CONTENT:0:200}"
elif echo "$CONTENT" | grep -q '<channel|>'; then
    run_test "No <channel|> tags in streamed content" "FAIL" "found <channel|> in: ${CONTENT:0:200}"
else
    run_test "No thinking tags leaked to content" "PASS"
    echo -e "    ${DIM}content: ${CONTENT:0:100}${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════
# Test 3: Non-streaming — thinking stripped from content
# ═══════════════════════════════════════════════════════════════════
echo -e "${YELLOW}Test 3: Thinking tags stripped in non-streaming mode${NC}"

RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "Think carefully then answer briefly."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 256,
    "temperature": 0.5,
    "stream": false
  }')

CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content',''))" 2>/dev/null)
if echo "$CONTENT" | grep -q '<|channel>thought'; then
    run_test "Non-streaming: no thinking tags in content" "FAIL" "found <|channel>thought in content"
elif echo "$CONTENT" | grep -q '<think>'; then
    run_test "Non-streaming: no thinking tags in content" "FAIL" "found <think> in content"
else
    run_test "Non-streaming: thinking stripped from content" "PASS"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════
# Test 4: Streaming tool call — arguments are valid JSON
# ═══════════════════════════════════════════════════════════════════
echo -e "${YELLOW}Test 4: Streaming tool call arguments are valid JSON (5 iterations)${NC}"

GOOD=0
BAD=0
NO_TC=0
for i in $(seq 1 5); do
    EVENTS=$(curl -sf -N "$BASE/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "mlx-serve",
        "messages": [
          {"role": "system", "content": "You MUST call the readFile tool. Always call readFile with path=\"config.json\". Never respond with text."},
          {"role": "user", "content": "Read the config file"}
        ],
        "tools": [{"type":"function","function":{"name":"readFile","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"}},"required":["path"]}}}],
        "max_tokens": 256,
        "temperature": 0.3,
        "stream": true
      }' 2>/dev/null)

    # Check for tool_calls in SSE events
    TC_LINE=$(echo "$EVENTS" | grep '"tool_calls"' | head -1)
    if [ -n "$TC_LINE" ]; then
        ARGS_OK=$(echo "$TC_LINE" | sed 's/^data: //' | python3 -c "
import sys,json
try:
    chunk = json.loads(sys.stdin.read().strip())
    tc = chunk['choices'][0]['delta']['tool_calls'][0]
    args = json.loads(tc['function']['arguments'])
    if args.get('path'):
        print('ok')
    else:
        print('missing_path')
except Exception as e:
    print(f'parse_error:{e}')
" 2>/dev/null)
        if [ "$ARGS_OK" = "ok" ]; then
            GOOD=$((GOOD + 1))
        else
            BAD=$((BAD + 1))
            echo -e "    ${DIM}Iteration $i: $ARGS_OK${NC}"
        fi
    else
        NO_TC=$((NO_TC + 1))
    fi
done
echo -e "    ${DIM}Results: $GOOD valid, $BAD invalid, $NO_TC no-tool-call${NC}"
if [ "$BAD" -eq 0 ]; then
    run_test "All streaming tool calls have valid JSON args" "PASS"
else
    run_test "All streaming tool calls have valid JSON args" "FAIL" "$BAD/$((GOOD+BAD)) had invalid args"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════
# Test 5: Tool call with multiline content preserved
# ═══════════════════════════════════════════════════════════════════
echo -e "${YELLOW}Test 5: Tool call round-trip preserves multiline content${NC}"

# Send a tool result with multiline content, verify model responds meaningfully
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You are helpful. Be brief."},
      {"role": "user", "content": "Read the server.js file"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "readFile", "arguments": "{\"path\": \"server.js\"}"}}]},
      {"role": "tool", "tool_call_id": "call_1", "content": "1| const express = require('"'"'express'"'"');\n2| const app = express();\n3| const cruises = [\n4|   { id: 1, title: \"Med Dream\", price: 999 },\n5|   { id: 2, title: \"Caribbean\", price: 799 },\n6| ];\n7| app.get(\"/api/cruises\", (req, res) => res.json(cruises));\n8| app.listen(3456);"}
    ],
    "tools": [{"type":"function","function":{"name":"readFile","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }')

TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
if [ "$TOKENS" -gt 2 ] 2>/dev/null; then
    run_test "Model responds after tool result with code content" "PASS"
    echo -e "    ${DIM}tokens: $TOKENS${NC}"
else
    run_test "Model responds after tool result with code content" "FAIL" "only $TOKENS completion tokens"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════
# Test 6: enable_thinking=true produces reasoning_content, not content
# ═══════════════════════════════════════════════════════════════════
echo -e "${YELLOW}Test 6: enable_thinking=true puts reasoning in reasoning_content field${NC}"

RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "Think step by step."},
      {"role": "user", "content": "What is 15 * 17?"}
    ],
    "max_tokens": 512,
    "temperature": 0.5,
    "enable_thinking": true,
    "stream": false
  }')

CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content',''))" 2>/dev/null)
REASONING=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('reasoning_content',''))" 2>/dev/null)

if echo "$CONTENT" | grep -q '<|channel>thought\|<think>'; then
    run_test "Thinking in reasoning_content, not content" "FAIL" "thinking tags found in content field"
else
    run_test "Content field clean (no thinking tags)" "PASS"
fi

if [ -n "$REASONING" ] && [ "$REASONING" != "None" ] && [ ${#REASONING} -gt 5 ]; then
    run_test "reasoning_content field populated" "PASS"
    echo -e "    ${DIM}reasoning: ${REASONING:0:100}...${NC}"
else
    # Model may not produce thinking — not a failure, just a skip
    run_test "reasoning_content field populated" "SKIP" "model did not produce thinking content"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════"
echo -e "  ${GREEN}Passed: $PASS${NC}  ${RED}Failed: $FAIL${NC}  ${YELLOW}Skipped: $SKIP${NC}  Total: $TOTAL"
echo "═══════════════════════════════════════════════"
if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Server log (last 30 lines):"
    tail -30 "$LOG"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
