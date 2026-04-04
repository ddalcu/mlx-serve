#!/bin/bash
# Full agent simulation with native tool calling
# Mimics exactly what the Swift app does
set -uo pipefail
PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0

pass() { PASS=$((PASS+1)); echo "✅ PASS: $1"; }
fail() { FAIL=$((FAIL+1)); echo "❌ FAIL: $1 — $2"; }

TOOLS='[
  {"type":"function","function":{"name":"shell","description":"Run a shell command on macOS and return the output.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute"}},"required":["command"]}}},
  {"type":"function","function":{"name":"writeFile","description":"Create or overwrite a file with the given content.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path to write to"},"content":{"type":"string","description":"The file content"}},"required":["path","content"]}}},
  {"type":"function","function":{"name":"readFile","description":"Read the contents of a file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path to read"}},"required":["path"]}}},
  {"type":"function","function":{"name":"editFile","description":"Find and replace text in an existing file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path to edit"},"find":{"type":"string","description":"Text to find"},"replace":{"type":"string","description":"Text to replace with"}},"required":["path","find","replace"]}}},
  {"type":"function","function":{"name":"searchFiles","description":"Search for a text pattern in files using grep.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Search pattern"},"path":{"type":"string","description":"Directory to search in"}},"required":["pattern"]}}},
  {"type":"function","function":{"name":"browse","description":"Navigate to a URL in the built-in browser and read the page content.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["navigate","readText","click","getInfo","executeJS"],"description":"Browser action"},"url":{"type":"string","description":"URL to navigate to"}},"required":["action"]}}},
  {"type":"function","function":{"name":"webSearch","description":"Search the web using DuckDuckGo and return the results as text.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}}}
]'

SYSTEM="You are a helpful macOS assistant. Use the provided tools when the user asks you to perform tasks. To search the web, use the webSearch tool."

call_model() {
    local messages="$1"
    local result=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"mlx-serve\",\"messages\":$messages,\"tools\":$TOOLS,\"max_tokens\":2048,\"stream\":false}" 2>&1)
    echo "$result"
}

extract_tool_calls() {
    python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
c = d['choices'][0]
fr = c.get('finish_reason','')
msg = c.get('message',{})
content = msg.get('content')
tcs = msg.get('tool_calls')
usage = d.get('usage',{})
print(f'FINISH: {fr}')
print(f'TOKENS: {usage.get(\"prompt_tokens\",0)}+{usage.get(\"completion_tokens\",0)}')
if content:
    print(f'CONTENT: {content[:200]}')
if tcs:
    for tc in tcs:
        f = tc['function']
        print(f'TOOL_CALL: {tc[\"id\"]}|{f[\"name\"]}|{f[\"arguments\"]}')
"
}

echo "=== MLX Serve Tool Calling Test Suite ==="
echo "Target: $BASE"
echo ""

# Test 1: Simple file creation
echo "--- Test 1: Create a file ---"
RESULT=$(call_model "[{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"Create a file called test.txt with the content Hello World\"}]")
INFO=$(echo "$RESULT" | extract_tool_calls)
echo "$INFO"
if echo "$INFO" | grep -q "TOOL_CALL.*writeFile"; then
    pass "create_file"
else
    fail "create_file" "no writeFile tool call"
fi

echo ""

# Test 2: Run a shell command
echo "--- Test 2: Shell command ---"
RESULT=$(call_model "[{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"List files in the current directory\"}]")
INFO=$(echo "$RESULT" | extract_tool_calls)
echo "$INFO"
if echo "$INFO" | grep -q "TOOL_CALL.*shell"; then
    pass "shell_command"
else
    fail "shell_command" "no shell tool call"
fi

echo ""

# Test 3: Normal question (no tools)
echo "--- Test 3: Normal question ---"
RESULT=$(call_model "[{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"What is 2+2?\"}]")
INFO=$(echo "$RESULT" | extract_tool_calls)
echo "$INFO"
if echo "$INFO" | grep -q "CONTENT:"; then
    if ! echo "$INFO" | grep -q "TOOL_CALL"; then
        pass "normal_question"
    else
        fail "normal_question" "used tools for a simple question"
    fi
else
    fail "normal_question" "no content"
fi

echo ""

# Test 4: Multi-turn with tool results
echo "--- Test 4: Multi-turn tool execution ---"
# Step 1: Ask to create and verify a file
RESULT=$(call_model "[{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"Create hello.html with a simple HTML page that says Hello\"}]")
INFO=$(echo "$RESULT" | extract_tool_calls)
echo "  Turn 1: $(echo "$INFO" | head -3)"
TC_ID=$(echo "$INFO" | grep "TOOL_CALL" | head -1 | cut -d'|' -f1 | sed 's/TOOL_CALL: //')
TC_NAME=$(echo "$INFO" | grep "TOOL_CALL" | head -1 | cut -d'|' -f2)

if [ -n "$TC_ID" ] && [ -n "$TC_NAME" ]; then
    # Step 2: Send tool result back
    TOOL_RESULT="File hello.html created with 150 characters."
    RESULT2=$(call_model "[{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"Create hello.html with a simple HTML page that says Hello\"},{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"$TC_ID\",\"type\":\"function\",\"function\":{\"name\":\"$TC_NAME\",\"arguments\":\"{\\\"path\\\":\\\"hello.html\\\",\\\"content\\\":\\\"<html><body><h1>Hello</h1></body></html>\\\"}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"$TC_ID\",\"content\":\"$TOOL_RESULT\"},{\"role\":\"user\",\"content\":\"Now read the file back to verify\"}]")
    INFO2=$(echo "$RESULT2" | extract_tool_calls)
    echo "  Turn 2: $(echo "$INFO2" | head -3)"
    if echo "$INFO2" | grep -q "TOOL_CALL.*readFile"; then
        pass "multi_turn_tool"
    elif echo "$INFO2" | grep -q "CONTENT:"; then
        echo "  (Model responded with text instead of reading file — acceptable)"
        pass "multi_turn_tool"
    else
        fail "multi_turn_tool" "unexpected response"
    fi
else
    fail "multi_turn_tool" "no tool call in turn 1"
fi

echo ""

# Test 5: Web search
echo "--- Test 5: Web search ---"
RESULT=$(call_model "[{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"Search the web for the latest news about Apple\"}]")
INFO=$(echo "$RESULT" | extract_tool_calls)
echo "$INFO"
if echo "$INFO" | grep -q "TOOL_CALL.*webSearch\|TOOL_CALL.*browse"; then
    pass "web_search"
else
    fail "web_search" "no search/browse tool call"
fi

echo ""

# Test 6: Complex multi-step task
echo "--- Test 6: Complex task ---"
RESULT=$(call_model "[{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"Check what Python version is installed\"}]")
INFO=$(echo "$RESULT" | extract_tool_calls)
echo "$INFO"
if echo "$INFO" | grep -q "TOOL_CALL.*shell"; then
    ARGS=$(echo "$INFO" | grep "TOOL_CALL" | cut -d'|' -f3)
    echo "  Command: $ARGS"
    pass "complex_task"
else
    fail "complex_task" "no shell tool call"
fi

echo ""

# Test 7: Tool call with large content (HTML page)
echo "--- Test 7: Large content creation ---"
RESULT=$(call_model "[{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"Create an HTML page about JFK called jfk.html with proper CSS styling\"}]")
INFO=$(echo "$RESULT" | extract_tool_calls)
echo "  $(echo "$INFO" | head -2)"
if echo "$INFO" | grep -q "TOOL_CALL.*writeFile"; then
    ARGS=$(echo "$INFO" | grep "TOOL_CALL" | cut -d'|' -f3)
    CONTENT_LEN=$(echo "$ARGS" | python3 -c "import sys,json; a=json.loads(sys.stdin.read()); print(len(a.get('content','')))" 2>/dev/null || echo "0")
    echo "  Content length: $CONTENT_LEN chars"
    if [ "$CONTENT_LEN" -gt 50 ]; then
        pass "large_content"
    else
        fail "large_content" "content too short: $CONTENT_LEN chars"
    fi
else
    fail "large_content" "no writeFile tool call"
fi

echo ""

# Test 8: Streaming with tools
echo "--- Test 8: Streaming with tools ---"
RESP=$(curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"system\",\"content\":\"$SYSTEM\"},{\"role\":\"user\",\"content\":\"Create a file called greeting.txt with Hi there\"}],\"tools\":$TOOLS,\"max_tokens\":256,\"stream\":true}" 2>&1)

if echo "$RESP" | grep -q "data: \[DONE\]"; then
    # Check if tool_calls appear in the stream
    if echo "$RESP" | grep -q "tool_calls"; then
        echo "  Streaming with tool_calls detected"
        pass "streaming_tools"
    else
        echo "  Stream completed but no tool_calls in stream"
        fail "streaming_tools" "no tool_calls in stream"
    fi
else
    fail "streaming_tools" "stream didn't complete"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $FAIL
