#!/bin/bash
# Integration test: reproduces "The model couldn't generate a response" with Gemma 4 tool calling.
# This sends the exact message sequence the agent loop produces after a webSearch tool call.
# Expected: model generates >2 completion tokens. Fails if completion_tokens <= 2.

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"

echo "=== Tool Response Integration Test ==="
echo "Server: $BASE"

# Check server is running
if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo "SKIP: Server not running on port $PORT"
    exit 0
fi

MODEL=$(curl -sf "$BASE/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
echo "Model: $MODEL"
echo ""

# Test 1: Baseline — simple chat without tools
echo "--- Test 1: Baseline chat (no tools) ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You are helpful. Be brief."},
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": false
  }')
TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:200])" 2>/dev/null)
echo "  completion_tokens: $TOKENS"
echo "  content: '$CONTENT'"
CLEANED=$(echo "$CONTENT" | sed 's/<pad>//g' | xargs)
if [ -n "$CLEANED" ]; then
    echo "  PASS (model generates non-empty content)"
else
    echo "  FAIL: baseline chat produces empty content"
    exit 1
fi
echo ""

# Test 2: Chat with tools param but no tool_calls in history
echo "--- Test 2: Chat with tools param (no tool history) ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You are helpful. Be brief."},
      {"role": "user", "content": "What is 2+2? Answer directly, no tools needed."}
    ],
    "tools": [{"type":"function","function":{"name":"shell","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": false
  }')
TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(m.get('content','')[:200])" 2>/dev/null)
echo "  completion_tokens: $TOKENS"
echo "  content: '$CONTENT'"
if [ -n "$(echo "$CONTENT" | sed 's/<pad>//g' | xargs)" ]; then
    echo "  PASS"
else
    echo "  FAIL: tools param without history produces empty content"
fi
echo ""

# Test 3: THE BUG — tool_calls in assistant + tool response + tools param
# This is the exact sequence that produces 2 tokens and [stop]
echo "--- Test 3: Tool call round-trip (THE BUG) ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You are helpful. Be brief."},
      {"role": "user", "content": "What day is today?"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_99_0", "type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"date\"}"}}]},
      {"role": "tool", "tool_call_id": "call_99_0", "content": "Fri Apr 4 16:30:00 PDT 2026"}
    ],
    "tools": [{"type":"function","function":{"name":"shell","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }')
TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
FINISH=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null)
CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(m.get('content','')[:200])" 2>/dev/null)
PROMPT_TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null)
echo "  prompt_tokens: $PROMPT_TOKENS"
echo "  completion_tokens: $TOKENS"
echo "  finish_reason: $FINISH"
echo "  content: '$CONTENT'"
if [ "$TOKENS" -gt 2 ] 2>/dev/null; then
    echo "  PASS: model generated meaningful response after tool result"
else
    echo "  FAIL: model generated <=2 tokens after tool result (finish=$FINISH)"
    echo "  This is the exact bug that causes 'The model couldn't generate a response.'"
    echo ""
    echo "  The Gemma 4 Jinja template likely rendered an invalid prompt."
    echo "  Dumping the raw response for debugging:"
    echo "$RESULT" | python3 -m json.tool 2>/dev/null
    FAILED_T3=1
fi
echo ""

# Test 4: Same conversation but WITHOUT tools param (no tool formatting)
echo "--- Test 4: Same messages but no tools param ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You are helpful. Be brief."},
      {"role": "user", "content": "What day is today?"},
      {"role": "assistant", "content": "Let me check. The date is Fri Apr 4 2026."},
      {"role": "user", "content": "Great, thanks! What year is it?"}
    ],
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": false
  }')
TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
echo "  completion_tokens: $TOKENS"
if [ "$TOKENS" -gt 2 ] 2>/dev/null; then
    echo "  PASS: model responds to multi-turn without tools"
else
    echo "  FAIL"
fi
echo ""

# Test 5: Tool response with role:"assistant" instead of role:"tool"
# This is what the Gemma fix should produce
echo "--- Test 5: Tool response as role:assistant (Gemma-native format) ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You are helpful. Be brief."},
      {"role": "user", "content": "What day is today?"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_99_0", "type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"date\"}"}}]},
      {"role": "assistant", "content": null, "tool_call_id": "call_99_0", "tool_responses": [{"name": "shell", "response": "Fri Apr 4 16:30:00 PDT 2026"}]}
    ],
    "tools": [{"type":"function","function":{"name":"shell","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }')
TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
FINISH=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null)
CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(m.get('content','')[:200])" 2>/dev/null)
echo "  completion_tokens: $TOKENS"
echo "  finish_reason: $FINISH"
echo "  content: '$CONTENT'"
if [ "$TOKENS" -gt 2 ] 2>/dev/null; then
    echo "  PASS: Gemma-native format works"
else
    echo "  FAIL: Gemma-native format also produces <=2 tokens"
    echo "$RESULT" | python3 -m json.tool 2>/dev/null
fi
echo ""

# Summary
echo "=== Summary ==="
if [ "$FAILED_T3" = "1" ]; then
    echo "Test 3 FAILED: Tool response causes <=2 tokens. This is the root cause."
    exit 1
else
    echo "All critical tests passed."
    exit 0
fi
