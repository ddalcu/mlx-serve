#!/bin/bash
# Integration tests for the Anthropic Messages API (/v1/messages).
# Tests both non-streaming and streaming, tool calling, thinking, and error handling.
# Usage: ./tests/test_anthropic_api.sh [port]
# Requires a running mlx-serve server with a loaded model.

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

assert_eq() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc"
        echo -e "    expected: $expected"
        echo -e "    actual:   $actual"
    fi
}

assert_contains() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q "$needle"; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc"
        echo -e "    expected to contain: $needle"
        echo -e "    actual: ${haystack:0:300}"
    fi
}

assert_not_contains() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q "$needle"; then
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc"
        echo -e "    should NOT contain: $needle"
    else
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc"
    fi
}

assert_gt() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" val="$2" threshold="$3"
    if [ "$val" -gt "$threshold" ] 2>/dev/null; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc ($val > $threshold)"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc ($val <= $threshold)"
    fi
}

echo "=== Anthropic Messages API Integration Tests ==="
echo "Server: $BASE"

# Check server is running
if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo "SKIP: Server not running on port $PORT"
    exit 0
fi

echo ""

# ── Test 1: Basic non-streaming message ──
echo "--- Test 1: Basic non-streaming /v1/messages ---"
RESULT=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: test-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "Say hello in one word."}
    ]
  }')

TYPE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['type'])" 2>/dev/null)
assert_eq "response type is message" "message" "$TYPE"

ROLE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['role'])" 2>/dev/null)
assert_eq "role is assistant" "assistant" "$ROLE"

STOP=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['stop_reason'])" 2>/dev/null)
assert_contains "stop_reason is end_turn or max_tokens" "end_turn\|max_tokens" "$STOP"

CONTENT_TYPE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['content'][0]['type'])" 2>/dev/null)
assert_eq "content block type is text" "text" "$CONTENT_TYPE"

CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['content'][0]['text'])" 2>/dev/null)
assert_gt "content has text" "${#CONTENT}" 0

INPUT_TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['input_tokens'])" 2>/dev/null)
assert_gt "input_tokens > 0" "$INPUT_TOKENS" 0

OUTPUT_TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['output_tokens'])" 2>/dev/null)
assert_gt "output_tokens > 0" "$OUTPUT_TOKENS" 0

MSG_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
assert_contains "id starts with msg_" "^msg_" "$MSG_ID"

echo ""

# ── Test 2: System prompt at top level ──
echo "--- Test 2: System prompt (top-level) ---"
RESULT=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 64,
    "system": "You are a pirate. Every response must include the word arrr.",
    "messages": [
      {"role": "user", "content": "Greet me."}
    ]
  }')
CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['content'][0]['text'])" 2>/dev/null)
assert_gt "system prompt produces response" "${#CONTENT}" 0
echo "  content: '${CONTENT:0:100}'"
echo ""

# ── Test 3: Streaming ──
echo "--- Test 3: Streaming /v1/messages ---"
SSE=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 32,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Say hi."}
    ]
  }')

assert_contains "has message_start event" "message_start" "$SSE"
assert_contains "has content_block_start event" "content_block_start" "$SSE"
assert_contains "has text_delta" "text_delta" "$SSE"
assert_contains "has content_block_stop event" "content_block_stop" "$SSE"
assert_contains "has message_delta event" "message_delta" "$SSE"
assert_contains "has message_stop event" "message_stop" "$SSE"
assert_contains "has stop_reason" "stop_reason" "$SSE"
assert_contains "has output_tokens in usage" "output_tokens" "$SSE"

# Verify message_start has correct structure
MSG_START=$(echo "$SSE" | grep "^data:.*message_start" | head -1 | sed 's/^data: //')
assert_contains "message_start has role:assistant" "assistant" "$MSG_START"
assert_contains "message_start has input_tokens" "input_tokens" "$MSG_START"

echo ""

# ── Test 4: Multi-turn conversation ──
echo "--- Test 4: Multi-turn conversation ---"
RESULT=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "My name is Alice."},
      {"role": "assistant", "content": [{"type": "text", "text": "Nice to meet you, Alice!"}]},
      {"role": "user", "content": "What is my name?"}
    ]
  }')
CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['content'][0]['text'])" 2>/dev/null)
assert_gt "multi-turn produces response" "${#CONTENT}" 0
echo "  content: '${CONTENT:0:100}'"
echo ""

# ── Test 5: Tool calling (non-streaming) ──
echo "--- Test 5: Tool calling (non-streaming) ---"
RESULT=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 256,
    "system": "You have access to tools. When the user asks what time or date it is, always use the shell tool to run the date command. Do not answer from memory.",
    "messages": [
      {"role": "user", "content": "What is the current date and time? Use the shell tool."}
    ],
    "tools": [
      {
        "name": "shell",
        "description": "Execute a shell command and return stdout",
        "input_schema": {
          "type": "object",
          "properties": {
            "command": {"type": "string", "description": "The shell command to run"}
          },
          "required": ["command"]
        }
      }
    ],
    "tool_choice": {"type": "any"}
  }')

STOP=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['stop_reason'])" 2>/dev/null)
# Model might respond with text or tool_use depending on model capability
HAS_TOOL_USE=$(echo "$RESULT" | python3 -c "
import sys,json
d = json.load(sys.stdin)
blocks = d.get('content', [])
has = any(b.get('type') == 'tool_use' for b in blocks)
print('yes' if has else 'no')
" 2>/dev/null)

echo "  stop_reason: $STOP"
echo "  has_tool_use: $HAS_TOOL_USE"

if [ "$HAS_TOOL_USE" = "yes" ]; then
    TC_NAME=$(echo "$RESULT" | python3 -c "
import sys,json
d = json.load(sys.stdin)
for b in d['content']:
    if b['type'] == 'tool_use':
        print(b['name']); break
" 2>/dev/null)
    TC_ID=$(echo "$RESULT" | python3 -c "
import sys,json
d = json.load(sys.stdin)
for b in d['content']:
    if b['type'] == 'tool_use':
        print(b['id']); break
" 2>/dev/null)
    TC_INPUT=$(echo "$RESULT" | python3 -c "
import sys,json
d = json.load(sys.stdin)
for b in d['content']:
    if b['type'] == 'tool_use':
        print(json.dumps(b['input'])); break
" 2>/dev/null)

    assert_eq "tool name is shell" "shell" "$TC_NAME"
    assert_contains "tool_use id starts with toolu_" "^toolu_" "$TC_ID"
    assert_contains "tool input has command" "command" "$TC_INPUT"
    assert_eq "stop_reason is tool_use" "tool_use" "$STOP"
else
    echo -e "  ${YELLOW}SKIP${NC} Model did not produce tool_use (model-dependent)"
fi
echo ""

# ── Test 6: Tool result round-trip ──
echo "--- Test 6: Tool result round-trip ---"
RESULT=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "What day is it?"},
      {"role": "assistant", "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_test_1", "name": "shell", "input": {"command": "date"}}
      ]},
      {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_test_1", "content": "Mon Apr 7 2026"}
      ]}
    ],
    "tools": [
      {
        "name": "shell",
        "description": "Execute a shell command",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
      }
    ]
  }')

OUTPUT_TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['output_tokens'])" 2>/dev/null)
assert_gt "tool result round-trip produces output" "$OUTPUT_TOKENS" 2

CONTENT=$(echo "$RESULT" | python3 -c "
import sys,json
d = json.load(sys.stdin)
for b in d['content']:
    if b['type'] == 'text':
        print(b['text'][:200]); break
" 2>/dev/null)
echo "  output_tokens: $OUTPUT_TOKENS"
echo "  content: '${CONTENT:0:100}'"
echo ""

# ── Test 7: Streaming with tools ──
echo "--- Test 7: Streaming with tool_choice=any ---"
SSE=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 256,
    "stream": true,
    "system": "Always use the shell tool when asked about time or date.",
    "messages": [
      {"role": "user", "content": "Run the date command using the shell tool."}
    ],
    "tools": [
      {
        "name": "shell",
        "description": "Execute a shell command",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
      }
    ],
    "tool_choice": {"type": "any"}
  }')

assert_contains "streaming has message_start" "message_start" "$SSE"
assert_contains "streaming has message_stop" "message_stop" "$SSE"

HAS_TOOL_STREAM=$(echo "$SSE" | grep -c "tool_use" || true)
if [ "$HAS_TOOL_STREAM" -gt 0 ]; then
    assert_contains "streaming tool has input_json_delta" "input_json_delta" "$SSE"
    assert_contains "streaming stop_reason is tool_use" "\"stop_reason\":\"tool_use\"" "$SSE"
else
    echo -e "  ${YELLOW}SKIP${NC} Model did not produce tool_use in streaming (model-dependent)"
fi
echo ""

# ── Test 8: Error handling — missing max_tokens ──
echo "--- Test 8: Error — missing max_tokens ---"
RESULT=$(curl -s "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [{"role": "user", "content": "hi"}]
  }')
ERR_TYPE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['error']['type'])" 2>/dev/null)
assert_eq "error type is invalid_request_error" "invalid_request_error" "$ERR_TYPE"
assert_contains "error mentions max_tokens" "max_tokens" "$RESULT"
echo ""

# ── Test 9: Error — missing messages ──
echo "--- Test 9: Error — missing messages ---"
RESULT=$(curl -s "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-serve", "max_tokens": 64}')
ERR_TYPE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['error']['type'])" 2>/dev/null)
assert_eq "error type is invalid_request_error" "invalid_request_error" "$ERR_TYPE"
echo ""

# ── Test 10: Error — invalid JSON ──
echo "--- Test 10: Error — invalid JSON ---"
RESULT=$(curl -s "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d 'not json')
assert_contains "error response for bad JSON" "invalid_request_error" "$RESULT"
echo ""

# ── Test 11: Anthropic headers accepted ──
echo "--- Test 11: Anthropic headers accepted ---"
RESULT=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-test-key" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: interleaved-thinking-2025-05-14" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}]
  }')
TYPE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['type'])" 2>/dev/null)
assert_eq "accepts Anthropic headers" "message" "$TYPE"
echo ""

# ── Test 12: stop_sequences ──
echo "--- Test 12: stop_sequences ---"
RESULT=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
    "stop_sequences": ["5"]
  }')
STOP=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['stop_reason'])" 2>/dev/null)
# Model may or may not hit the stop sequence depending on output format
echo "  stop_reason: $STOP"
OUTPUT_TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['output_tokens'])" 2>/dev/null)
assert_gt "stop_sequences: produces some output" "$OUTPUT_TOKENS" 0
echo ""

# ── Test 13: Model name echoed back ──
echo "--- Test 13: Model name echoed back ---"
RESULT=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-opus-4-6",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}]
  }')
MODEL=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['model'])" 2>/dev/null)
assert_eq "model name echoed back" "claude-opus-4-6" "$MODEL"
echo ""

# ── Test 14: Streaming event order ──
echo "--- Test 14: Streaming event order ---"
SSE=$(curl -sf "$BASE/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "max_tokens": 16,
    "stream": true,
    "messages": [{"role": "user", "content": "Say one word."}]
  }')

# Extract event types in order
EVENTS=$(echo "$SSE" | grep "^event:" | sed 's/event: //')
FIRST_EVENT=$(echo "$EVENTS" | head -1)
LAST_EVENT=$(echo "$EVENTS" | tail -1)
assert_eq "first event is message_start" "message_start" "$FIRST_EVENT"
assert_eq "last event is message_stop" "message_stop" "$LAST_EVENT"

# Check content_block_start appears before content_block_delta
BLOCK_START_LINE=$(echo "$EVENTS" | grep -n "content_block_start" | head -1 | cut -d: -f1)
DELTA_LINE=$(echo "$EVENTS" | grep -n "content_block_delta" | head -1 | cut -d: -f1)
if [ -n "$BLOCK_START_LINE" ] && [ -n "$DELTA_LINE" ]; then
    assert_gt "content_block_start before delta" "$DELTA_LINE" "$BLOCK_START_LINE"
fi

# Check message_delta appears before message_stop
MSG_DELTA_LINE=$(echo "$EVENTS" | grep -n "message_delta" | head -1 | cut -d: -f1)
MSG_STOP_LINE=$(echo "$EVENTS" | grep -n "message_stop" | head -1 | cut -d: -f1)
if [ -n "$MSG_DELTA_LINE" ] && [ -n "$MSG_STOP_LINE" ]; then
    assert_gt "message_delta before message_stop" "$MSG_STOP_LINE" "$MSG_DELTA_LINE"
fi
echo ""

# ── Summary ──
echo "=== Summary ==="
echo -e "Total: $TOTAL  ${GREEN}Passed: $PASS${NC}  ${RED}Failed: $FAIL${NC}"
if [ $FAIL -gt 0 ]; then
    exit 1
else
    echo "All Anthropic API tests passed."
    exit 0
fi
