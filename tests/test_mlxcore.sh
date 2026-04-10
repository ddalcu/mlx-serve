#!/bin/bash
# End-to-end test for MLX Core app via AppleScript UI automation + API monitoring.
#
# Requires:
#   - MLX Core app running with a model loaded
#   - Accessibility permissions for Terminal (System Settings → Privacy → Accessibility)
#
# What it does:
#   1. Opens a chat window in MLX Core
#   2. Enables agent mode
#   3. Types test prompts and sends them
#   4. Monitors server logs, chat history, and API requests
#   5. Validates tool calls and responses
#
# Usage: ./tests/test_mlxcore.sh [port]

set -uo pipefail

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"
HISTORY="$HOME/.mlx-serve/chat-history.json"
LAST_REQ="$HOME/.mlx-serve/last-agent-request.json"
LOG_FILE="/tmp/mlxcore-test.log"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'
PASS=0
FAIL=0

> "$LOG_FILE"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
pass() { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1" | tee -a "$LOG_FILE"; }
fail() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1" | tee -a "$LOG_FILE"; [ -n "${2:-}" ] && echo "    $2" | tee -a "$LOG_FILE"; }

echo "═══════════════════════════════════════════════"
echo "  MLX Core End-to-End Test"
echo "═══════════════════════════════════════════════"
echo ""

# ── Check prerequisites ──
log "Checking prerequisites..."

# App running?
if ! pgrep -q MLXCore; then
    echo -e "${RED}MLX Core is not running. Launch it first.${NC}"
    exit 1
fi
pass "MLX Core app is running"

# Server running?
if curl -sf "$BASE/health" > /dev/null 2>&1; then
    pass "Server is healthy on port $PORT"
else
    fail "Server not responding on port $PORT" "Start a model in MLX Core first"
    exit 1
fi

# Check accessibility
if osascript -e 'tell application "System Events" to get name of process "MLXCore"' &>/dev/null; then
    pass "Accessibility access granted"
    HAS_ACCESSIBILITY=true
else
    fail "Accessibility access not granted" "Grant Terminal accessibility in System Settings → Privacy → Accessibility"
    echo -e "${YELLOW}Falling back to API-only testing (no UI automation)${NC}"
    HAS_ACCESSIBILITY=false
fi

# ── Helper: get message count from chat history ──
get_message_count() {
    python3 -c "
import json
with open('$HISTORY') as f:
    sessions = json.load(f)
if sessions:
    print(len(sessions[0].get('messages', [])))
else:
    print(0)
" 2>/dev/null || echo 0
}

# ── Helper: get last assistant message ──
get_last_assistant() {
    python3 -c "
import json
with open('$HISTORY') as f:
    sessions = json.load(f)
if sessions:
    msgs = sessions[0].get('messages', [])
    for m in reversed(msgs):
        if m.get('role') == 'assistant':
            content = m.get('content', '')
            tc = m.get('toolCalls', [])
            if tc:
                print('TOOL_CALLS:' + json.dumps(tc))
            elif content:
                print('CONTENT:' + content[:200])
            else:
                print('EMPTY')
            break
    else:
        print('NONE')
else:
    print('NONE')
" 2>/dev/null || echo "ERROR"
}

# ── Helper: wait for new message in chat history ──
wait_for_response() {
    local initial_count="$1"
    local timeout="${2:-120}"
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        local current
        current=$(get_message_count)
        if [ "$current" -gt "$initial_count" ]; then
            # Wait a bit more for streaming to finish
            sleep 3
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

# ── Helper: send chat via AppleScript ──
send_chat_message() {
    local message="$1"
    if [ "$HAS_ACCESSIBILITY" = "true" ]; then
        osascript <<EOF
tell application "MLX Core" to activate
delay 0.5
tell application "System Events"
    tell process "MLXCore"
        -- Click on the text field (chat input)
        keystroke "$message"
        delay 0.3
        -- Press Enter to send
        key code 36
    end tell
end tell
EOF
    else
        log "  (no accessibility — skipping UI send for: $message)"
        return 1
    fi
}

# ── Helper: create new chat session via AppleScript ──
new_chat_session() {
    if [ "$HAS_ACCESSIBILITY" = "true" ]; then
        osascript <<EOF
tell application "MLX Core" to activate
delay 0.5
tell application "System Events"
    tell process "MLXCore"
        -- Cmd+N for new chat
        keystroke "n" using command down
        delay 0.5
    end tell
end tell
EOF
    fi
}

# ── Monitor: tail server log from app ──
echo ""
log "Starting server log monitor..."
# The server log is captured in the app's ServerManager.serverLog
# We can also monitor via direct API calls

# ── Test: API-based agent loop (works without accessibility) ──
echo ""
echo -e "${CYAN}━━━ Test 1: API agent loop (simulating app behavior) ━━━${NC}"
log "Running API-based agent test..."

# Replicate exactly what the app sends
AGENT_SYSTEM="You are a helpful macOS assistant. Use tools for tasks. Answer directly when no tools needed. For web search use webSearch tool."
AGENT_TOOLS=$(python3 -c "
import json
tools = [
    {'type': 'function', 'function': {'name': 'shell', 'description': 'Run a command', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}},
    {'type': 'function', 'function': {'name': 'writeFile', 'description': 'Write a file', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content']}}},
    {'type': 'function', 'function': {'name': 'readFile', 'description': 'Read a file', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'browse', 'description': 'Browse a URL', 'parameters': {'type': 'object', 'properties': {'action': {'type': 'string'}, 'url': {'type': 'string'}}, 'required': ['action']}}},
    {'type': 'function', 'function': {'name': 'webSearch', 'description': 'Search the web', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}},
]
print(json.dumps(tools))
")

# Test: simple tool call via streaming (matching app's exact format)
BODY=$(python3 -c "
import json
body = {
    'model': 'mlx-serve',
    'messages': [
        {'role': 'system', 'content': '$AGENT_SYSTEM'},
        {'role': 'user', 'content': 'What is 2+2?'}
    ],
    'tools': json.loads('$AGENT_TOOLS'),
    'max_tokens': 2048,
    'temperature': 0.7,
    'top_p': 0.95,
    'stream': True,
    'stream_options': {'include_usage': True}
}
print(json.dumps(body))
")

RESP=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY" 2>/dev/null)
CONTENT=$(echo "$RESP" | python3 -c "
import sys
lines = sys.stdin.read().strip().split('\n')
parts = []
for line in lines:
    if line.startswith('data: ') and line.strip() != 'data: [DONE]':
        import json
        try:
            chunk = json.loads(line[6:])
            delta = chunk.get('choices', [{}])[0].get('delta', {})
            if delta.get('content'):
                parts.append(delta['content'])
        except: pass
print(''.join(parts))
" 2>/dev/null)

if echo "$CONTENT" | grep -qi "4"; then
    pass "API streaming: model answered 2+2 correctly"
else
    fail "API streaming: model answered 2+2" "got: ${CONTENT:0:100}"
fi

# Test: tool call via streaming
BODY2=$(python3 -c "
import json
body = {
    'model': 'mlx-serve',
    'messages': [
        {'role': 'system', 'content': '$AGENT_SYSTEM'},
        {'role': 'user', 'content': 'Search the web for latest AI news'}
    ],
    'tools': json.loads('$AGENT_TOOLS'),
    'max_tokens': 2048,
    'temperature': 0.7,
    'top_p': 0.95,
    'stream': True,
    'stream_options': {'include_usage': True}
}
print(json.dumps(body))
")

RESP2=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY2" 2>/dev/null)
HAS_TOOL=$(echo "$RESP2" | python3 -c "
import sys
lines = sys.stdin.read().strip().split('\n')
for line in lines:
    if line.startswith('data: ') and 'tool_calls' in line and line.strip() != 'data: [DONE]':
        print('yes')
        break
else:
    print('no')
" 2>/dev/null)

FINISH=$(echo "$RESP2" | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
for line in lines:
    if line.startswith('data: ') and line.strip() != 'data: [DONE]':
        try:
            chunk = json.loads(line[6:])
            fr = chunk.get('choices', [{}])[0].get('finish_reason')
            if fr: print(fr)
        except: pass
" 2>/dev/null | tail -1)

if [ "$FINISH" = "tool_calls" ]; then
    pass "API streaming: tool call detected (finish_reason=tool_calls)"
else
    fail "API streaming: tool call detected" "finish_reason=$FINISH"
fi

if [ "$HAS_TOOL" = "yes" ]; then
    pass "API streaming: tool_calls delta present in SSE"
else
    fail "API streaming: tool_calls delta present in SSE"
fi

# ── Test: UI automation (only if accessibility granted) ──
if [ "$HAS_ACCESSIBILITY" = "true" ]; then
    echo ""
    echo -e "${CYAN}━━━ Test 2: UI automation via AppleScript ━━━${NC}"

    # Create a new chat session
    new_chat_session
    sleep 1

    INITIAL_COUNT=$(get_message_count)
    log "Initial message count: $INITIAL_COUNT"

    # Send a simple message
    log "Sending: 'Hello, what can you do?'"
    send_chat_message "Hello, what can you do?"
    sleep 1

    if wait_for_response "$INITIAL_COUNT" 60; then
        LAST=$(get_last_assistant)
        if echo "$LAST" | grep -q "CONTENT:"; then
            pass "UI: model responded to chat message"
            PREVIEW=$(echo "$LAST" | sed 's/CONTENT://' | head -c 80)
            log "  Response: $PREVIEW..."
        else
            fail "UI: model responded with content" "got: $LAST"
        fi
    else
        fail "UI: response received within 60s" "timeout"
    fi

    # Test agent mode with tool call
    INITIAL_COUNT=$(get_message_count)
    log "Sending agent task: 'What time is it? Use the shell tool to run date'"
    send_chat_message "What time is it? Use the shell tool to run date"
    sleep 1

    if wait_for_response "$INITIAL_COUNT" 90; then
        # Check if the last-agent-request was dumped
        if [ -f "$LAST_REQ" ]; then
            HAS_TOOLS_IN_REQ=$(python3 -c "
import json
with open('$LAST_REQ') as f:
    req = json.load(f)
print('yes' if req.get('tools') else 'no')
" 2>/dev/null)
            if [ "$HAS_TOOLS_IN_REQ" = "yes" ]; then
                pass "UI: agent request includes tools"
            else
                fail "UI: agent request includes tools"
            fi
        fi

        # Wait for tool execution + final response
        sleep 10
        LAST=$(get_last_assistant)
        if echo "$LAST" | grep -qi "CONTENT:"; then
            pass "UI: agent completed task with response"
        elif echo "$LAST" | grep -q "TOOL_CALLS:"; then
            pass "UI: agent made tool calls"
        else
            fail "UI: agent response" "got: $LAST"
        fi
    else
        fail "UI: agent response within 90s" "timeout"
    fi
else
    echo ""
    echo -e "${YELLOW}━━━ Test 2: UI automation SKIPPED (no accessibility) ━━━${NC}"
    echo "  Grant accessibility to Terminal in System Settings to enable UI tests."
fi

# ── Test: Server health during load ──
echo ""
echo -e "${CYAN}━━━ Test 3: Server stability ━━━${NC}"

# Rapid health checks
HEALTH_OK=0
for i in $(seq 1 10); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then
        HEALTH_OK=$((HEALTH_OK + 1))
    fi
done
if [ $HEALTH_OK -eq 10 ]; then
    pass "Server stable: 10/10 health checks passed"
else
    fail "Server stable" "$HEALTH_OK/10 health checks passed"
fi

# Check server didn't crash
if pgrep -q mlx-serve; then
    pass "mlx-serve process still alive"
else
    fail "mlx-serve process still alive" "process died during testing"
fi

if pgrep -q MLXCore; then
    pass "MLXCore process still alive"
else
    fail "MLXCore process still alive" "app crashed during testing"
fi

# ── Summary ──
TOTAL=$((PASS + FAIL))
echo ""
echo "═══════════════════════════════════════════════"
echo "  Results: $TOTAL tests"
echo -e "  ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "  Log: $LOG_FILE"
echo "═══════════════════════════════════════════════"
exit $FAIL
