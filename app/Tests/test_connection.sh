#!/bin/bash
# Test suite for MLX Serve connection issues
# Tests the exact scenarios that cause "The network connection was lost" in the app

set -uo pipefail
PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0

pass() { PASS=$((PASS+1)); echo "✅ PASS: $1"; }
fail() { FAIL=$((FAIL+1)); echo "❌ FAIL: $1 — $2"; }

echo "=== MLX Serve Connection Test Suite ==="
echo "Target: $BASE"
echo ""

# Test 1: Basic health
echo "--- Test 1: Health Check ---"
RESP=$(curl -s -w "\n%{http_code}" "$BASE/health" 2>&1)
CODE=$(echo "$RESP" | tail -1)
if [ "$CODE" = "200" ]; then pass "health"; else fail "health" "HTTP $CODE"; fi

# Test 2: Models endpoint
echo "--- Test 2: Models ---"
RESP=$(curl -s -w "\n%{http_code}" "$BASE/v1/models" 2>&1)
CODE=$(echo "$RESP" | tail -1)
if [ "$CODE" = "200" ]; then pass "models"; else fail "models" "HTTP $CODE"; fi

# Test 3: Simple streaming chat
echo "--- Test 3: Simple Streaming Chat ---"
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"stream":true}' 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then pass "simple_stream"; else fail "simple_stream" "no DONE marker"; fi

# Test 4: Plist content (THE problematic text)
echo "--- Test 4: Plist Content ---"
PLIST_JSON=$(python3 -c "
import json
text = '''summarize this: <?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>CFBundleName</key>
    <string>MLX Core</string>
    <key>CFBundleDisplayName</key>
    <string>MLX Core</string>
    <key>CFBundleIdentifier</key>
    <string>com.dalcu.mlx-core</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleExecutable</key>
    <string>MLXCore</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsLocalNetworking</key>
        <true/>
    </dict>
    <key>NSAppleEventsUsageDescription</key>
    <string>MLX Core uses AppleScript to control Safari and other apps in agent mode.</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright 2026 David Dalcu. All rights reserved.</string>
</dict>
</plist>'''
body = {'model':'mlx-serve','messages':[{'role':'user','content':text}],'max_tokens':256,'stream':True,'temperature':0.8,'top_p':0.95}
print(json.dumps(body))
")
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d "$PLIST_JSON" 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then
    CONTENT=$(echo "$RESP" | grep "^data:" | head -3)
    echo "  First chunks: $(echo "$CONTENT" | head -2)"
    pass "plist_content"
else
    echo "  Response: $(echo "$RESP" | head -5)"
    fail "plist_content" "no DONE marker or error"
fi

# Test 5: Multi-turn conversation with growing history
echo "--- Test 5: Multi-Turn Chat ---"
HISTORY='[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there! How can I help you?"},{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4"},{"role":"user","content":"Thanks! Now say goodbye."}]'
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d "{\"model\":\"mlx-serve\",\"messages\":$HISTORY,\"max_tokens\":32,\"stream\":true}" 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then pass "multi_turn"; else fail "multi_turn" "no DONE marker"; fi

# Test 6: Multi-turn with plist injected
echo "--- Test 6: Multi-Turn with Plist ---"
MULTI_PLIST=$(python3 -c "
import json
plist = '''<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict><key>CFBundleName</key><string>Test</string></dict></plist>'''
msgs = [
    {'role':'user','content':'Hello'},
    {'role':'assistant','content':'Hi there!'},
    {'role':'user','content':'summarize this: '+plist}
]
print(json.dumps({'model':'mlx-serve','messages':msgs,'max_tokens':128,'stream':True}))
")
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d "$MULTI_PLIST" 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then pass "multi_turn_plist"; else fail "multi_turn_plist" "no DONE"; fi

# Test 7: Rapid sequential requests (connection reuse stress)
echo "--- Test 7: Rapid Sequential (5 requests) ---"
RAPID_OK=true
for i in 1 2 3 4 5; do
    RESP=$(curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Connection: close" \
      -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Say $i.\"}],\"max_tokens\":16,\"stream\":true}" 2>&1)
    if ! echo "$RESP" | grep -q "data: \[DONE\]"; then
        fail "rapid_$i" "failed"
        RAPID_OK=false
        break
    fi
    echo "  Rapid $i: OK"
done
if [ "$RAPID_OK" = true ]; then pass "rapid_sequential"; fi

# Test 8: Health during streaming (concurrent)
echo "--- Test 8: Health During Streaming ---"
# Start a long streaming request in background
curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Write a very long essay about the history of computing."}],"max_tokens":256,"stream":true}' > /tmp/mlx_stream_test.txt 2>&1 &
STREAM_PID=$!
sleep 2

# Do health checks while streaming
HEALTH_OK=true
for i in 1 2 3; do
    H=$(curl -s --max-time 5 "$BASE/health" 2>&1)
    if ! echo "$H" | grep -q "ok"; then
        fail "health_during_stream_$i" "health check failed"
        HEALTH_OK=false
        break
    fi
    sleep 1
done
wait $STREAM_PID 2>/dev/null
if [ "$HEALTH_OK" = true ]; then
    if grep -q "data: \[DONE\]" /tmp/mlx_stream_test.txt; then
        pass "health_during_stream"
    else
        fail "health_during_stream" "stream didn't complete"
    fi
fi
rm -f /tmp/mlx_stream_test.txt

# Test 9: Non-streaming request
echo "--- Test 9: Non-Streaming ---"
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Say OK."}],"max_tokens":32,"stream":false}' 2>&1)
if echo "$RESP" | grep -q '"choices"'; then pass "non_streaming"; else fail "non_streaming" "unexpected response: $(echo "$RESP" | head -1)"; fi

# Test 10: Special characters (HTML, JSON, unicode, backslashes)
echo "--- Test 10: Special Characters ---"
SPECIAL_JSON=$(python3 -c "
import json
text = 'Summarize: <html><body><script>alert(\"xss\")</script></body></html>\nJSON: {\"key\": \"value\"}\nUnicode: 日本語 🎉\nBackslash: C:\\\\Users\\\\test'
print(json.dumps({'model':'mlx-serve','messages':[{'role':'user','content':text}],'max_tokens':64,'stream':True}))
")
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d "$SPECIAL_JSON" 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then pass "special_chars"; else fail "special_chars" "no DONE"; fi

# Test 11: Connection: keep-alive vs close behavior
echo "--- Test 11: Connection Header Behavior ---"
# With keep-alive (how URLSession used to send)
RESP=$(curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: keep-alive" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Say A."}],"max_tokens":16,"stream":true}' 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then pass "keepalive_header"; else fail "keepalive_header" "failed with keep-alive"; fi

# Immediately after with close
RESP=$(curl -s --max-time 60 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Say B."}],"max_tokens":16,"stream":true}' 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then pass "close_after_keepalive"; else fail "close_after_keepalive" "failed after keep-alive"; fi

# Test 12: Large request body (agent-mode system prompt)
echo "--- Test 12: Large System Prompt ---"
LARGE_PROMPT=$(python3 -c "
import json
system = 'You are an AI agent. ' * 200  # ~4KB system prompt
msgs = [{'role':'system','content':system},{'role':'user','content':'List files.'}]
print(json.dumps({'model':'mlx-serve','messages':msgs,'max_tokens':64,'stream':True}))
")
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d "$LARGE_PROMPT" 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then pass "large_system_prompt"; else fail "large_system_prompt" "no DONE: $(echo "$RESP" | head -2)"; fi

# Test 13: Thinking mode
echo "--- Test 13: Thinking Mode ---"
RESP=$(curl -s --max-time 120 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d '{"model":"mlx-serve","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":128,"stream":true,"enable_thinking":true}' 2>&1)
if echo "$RESP" | grep -q "data: \[DONE\]"; then
    if echo "$RESP" | grep -q "reasoning_content"; then
        echo "  Has reasoning content"
    else
        echo "  No reasoning content (model may not support it)"
    fi
    pass "thinking_mode"
else
    fail "thinking_mode" "no DONE"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $FAIL
