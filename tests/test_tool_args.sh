#!/bin/bash
# Integration test: verify tool call arguments are parsed correctly.
# Tests the fix for intermittent "missing parameter: command, args: [none]"
# caused by convertGemma4ArgsToJson producing invalid JSON for quoted keys.
#
# Usage: ./tests/test_tool_args.sh [model_dir] [port]
# Starts its own server instance, runs tests, then kills it.

MODEL_DIR=${1:-~/.mlx-serve/models/gemma-4-e4b-it-4bit}
PORT=${2:-8099}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

echo "=== Tool Call Arguments Integration Test ==="
echo "Model: $MODEL_DIR"
echo "Port: $PORT"
echo ""

# Start server
echo "Starting server..."
./zig-out/bin/mlx-serve --model "$MODEL_DIR" --serve --port $PORT --log-level debug 2>/tmp/mlx-serve-test-tool-args.log &
SERVER_PID=$!
sleep 2

# Wait for server to be ready
for i in $(seq 1 30); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then
        echo "Server ready (PID $SERVER_PID)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "FAIL: Server did not start within 30s"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done
echo ""

cleanup() {
    echo ""
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

run_test() {
    local name="$1"
    local result="$2"  # "PASS" or "FAIL"
    local detail="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $name"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name — $detail"
    fi
}

# ── Test 1: Non-streaming tool call with arguments ──
echo "--- Test 1: Non-streaming — model generates tool call with arguments ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You MUST use the shell tool for every request. Always call the shell tool. Never respond with text alone."},
      {"role": "user", "content": "Run the ls command to list files"}
    ],
    "tools": [{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Shell command to run"}},"required":["command"]}}}],
    "max_tokens": 512,
    "temperature": 0.3,
    "stream": false
  }')

FINISH=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null)
if [ "$FINISH" = "tool_calls" ]; then
    ARGS=$(echo "$RESULT" | python3 -c "
import sys,json
r = json.load(sys.stdin)
tc = r['choices'][0]['message']['tool_calls'][0]
args = json.loads(tc['function']['arguments'])
print(json.dumps(args))
" 2>/dev/null)
    if [ -n "$ARGS" ] && [ "$ARGS" != "{}" ] && [ "$ARGS" != "null" ]; then
        run_test "Non-streaming tool call has arguments" "PASS" ""
        echo "    arguments: $ARGS"
    else
        run_test "Non-streaming tool call has arguments" "FAIL" "arguments empty or null: $ARGS"
        echo "    raw response: $(echo "$RESULT" | python3 -m json.tool 2>/dev/null | head -20)"
    fi
else
    echo "  SKIP: model did not generate tool_calls (finish_reason=$FINISH)"
    echo "    (This is model-dependent, not a bug)"
fi
echo ""

# ── Test 2: Streaming tool call with arguments ──
echo "--- Test 2: Streaming — tool call arguments arrive correctly ---"
# Collect all SSE events
EVENTS=$(curl -sf -N "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You MUST use the shell tool for every request. Always call the shell tool. Never respond with text alone."},
      {"role": "user", "content": "List files using ls -la"}
    ],
    "tools": [{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Shell command to run"}},"required":["command"]}}}],
    "max_tokens": 512,
    "temperature": 0.3,
    "stream": true,
    "stream_options": {"include_usage": true}
  }' 2>/dev/null)

# Check if any event has tool_calls
HAS_TOOL_CALLS=$(echo "$EVENTS" | grep -o '"tool_calls"' | head -1)
if [ -n "$HAS_TOOL_CALLS" ]; then
    # Extract the tool call delta
    TC_LINE=$(echo "$EVENTS" | grep '"tool_calls"' | head -1)
    TC_ARGS=$(echo "$TC_LINE" | sed 's/^data: //' | python3 -c "
import sys,json
chunk = json.load(sys.stdin)
tc = chunk['choices'][0]['delta']['tool_calls'][0]
args_str = tc['function']['arguments']
# Verify it's valid JSON by parsing
args = json.loads(args_str)
print(json.dumps(args))
" 2>/dev/null)

    if [ -n "$TC_ARGS" ] && [ "$TC_ARGS" != "{}" ] && [ "$TC_ARGS" != "null" ]; then
        run_test "Streaming tool call has arguments" "PASS" ""
        echo "    arguments: $TC_ARGS"
    else
        run_test "Streaming tool call has arguments" "FAIL" "arguments empty or invalid"
        echo "    raw tool_calls line: $TC_LINE"
    fi

    # Also check finish_reason
    FINISH_LINE=$(echo "$EVENTS" | grep '"finish_reason":"tool_calls"')
    if [ -n "$FINISH_LINE" ]; then
        run_test "Streaming finish_reason is tool_calls" "PASS" ""
    else
        run_test "Streaming finish_reason is tool_calls" "FAIL" "no finish_reason=tool_calls found"
    fi
else
    echo "  SKIP: model did not generate tool_calls in streaming mode"
    echo "    (This is model-dependent, not a bug)"
fi
echo ""

# ── Test 3: Multiple iterations to catch intermittent failures ──
echo "--- Test 3: Repeat tool call test (5 iterations for intermittent bug) ---"
EMPTY_ARGS=0
GOOD_CALLS=0
NO_CALLS=0
for i in $(seq 1 5); do
    RESULT=$(curl -sf "$BASE/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "mlx-serve",
        "messages": [
          {"role": "system", "content": "Always use the shell tool. You must call shell for every request."},
          {"role": "user", "content": "Check disk usage with df -h"}
        ],
        "tools": [{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Shell command to run"}},"required":["command"]}}}],
        "max_tokens": 512,
        "temperature": 0.5,
        "stream": false
      }')

    FINISH=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['finish_reason'])" 2>/dev/null)
    if [ "$FINISH" = "tool_calls" ]; then
        ARGS=$(echo "$RESULT" | python3 -c "
import sys,json
r = json.load(sys.stdin)
tc = r['choices'][0]['message']['tool_calls'][0]
args = json.loads(tc['function']['arguments'])
if args:
    print('ok')
else:
    print('empty')
" 2>/dev/null)
        if [ "$ARGS" = "ok" ]; then
            GOOD_CALLS=$((GOOD_CALLS + 1))
        else
            EMPTY_ARGS=$((EMPTY_ARGS + 1))
            echo "    Iteration $i: EMPTY ARGS — $(echo "$RESULT" | python3 -c "import sys,json; tc=json.load(sys.stdin)['choices'][0]['message']['tool_calls'][0]; print(tc['function'])" 2>/dev/null)"
        fi
    else
        NO_CALLS=$((NO_CALLS + 1))
    fi
done
echo "  Results: $GOOD_CALLS good, $EMPTY_ARGS empty-args, $NO_CALLS no-tool-call"
if [ "$EMPTY_ARGS" -eq 0 ]; then
    run_test "No empty arguments in 5 iterations" "PASS" ""
else
    run_test "No empty arguments in 5 iterations" "FAIL" "$EMPTY_ARGS iterations had empty arguments"
fi
echo ""

# ── Test 4: Tool call round-trip (call → result → response) ──
echo "--- Test 4: Full round-trip — tool call then tool result then response ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "system", "content": "You are helpful. Be brief."},
      {"role": "user", "content": "What files are in the current directory?"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1_0", "type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"ls -la\"}"}}]},
      {"role": "tool", "tool_call_id": "call_1_0", "content": "total 16\ndrwxr-xr-x  4 user staff 128 Apr  6 10:00 .\n-rw-r--r--  1 user staff 100 Apr  6 10:00 README.md\n-rw-r--r--  1 user staff 200 Apr  6 10:00 main.py"}
    ],
    "tools": [{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Shell command to run"}},"required":["command"]}}}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }')
TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print(m.get('content','')[:200])" 2>/dev/null)
if [ "$TOKENS" -gt 2 ] 2>/dev/null; then
    run_test "Round-trip produces meaningful response" "PASS" ""
    echo "    tokens: $TOKENS, content: ${CONTENT:0:100}"
else
    run_test "Round-trip produces meaningful response" "FAIL" "only $TOKENS completion tokens"
fi
echo ""

# ── Summary ──
echo "=== Summary ==="
echo "Passed: $PASS / $TOTAL"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed: $FAIL"
    echo ""
    echo "Server log (last 50 lines):"
    tail -50 /tmp/mlx-serve-test-tool-args.log
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
