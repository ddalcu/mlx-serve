#!/bin/bash
# Comprehensive Swift agent harness test.
# Tests ALL tools through the app's TestServer API.
#
# Usage: ./tests/test_swift_agent.sh [test_server_port]
# Requires: running MLX Core app with a loaded model

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
WORKSPACE="$HOME/.mlx-serve/workspace/agent_test"
PASS=0
FAIL=0
TOTAL=0
TOOLLOG="$HOME/.mlx-serve/tool-calls.log"

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

# Verify server is up
STATUS=$(curl -sf "$BASE/test/status" 2>/dev/null)
if ! echo "$STATUS" | grep -q '"running"'; then
    echo "ERROR: TestServer not running or model not loaded on port $PORT"
    echo "$STATUS"
    exit 1
fi

# Setup
mkdir -p "$WORKSPACE"
rm -f "$WORKSPACE"/*.txt "$WORKSPACE"/*.py 2>/dev/null
> "$TOOLLOG"

run_agent() {
    local msg_file="$1" max_rounds="${2:-5}" timeout="${3:-120}"
    local msg
    msg=$(cat "$msg_file")
    # Reset session
    curl -sf "$BASE/test/reset" -X POST > /dev/null
    # Build JSON body with python to handle escaping
    local body
    body=$(python3 -c "
import json, sys
msg = sys.stdin.read()
print(json.dumps({'message': msg, 'max_rounds': $max_rounds, 'working_directory': '$WORKSPACE'}))
" <<< "$msg")
    # Start agent job
    curl -sf "$BASE/test/agent" \
        -H "Content-Type: application/json" \
        -d "$body" > /dev/null
    # Poll for completion
    for i in $(seq 1 $((timeout / 3))); do
        sleep 3
        RESULT=$(curl -sf "$BASE/test/agent/status" 2>/dev/null)
        local status
        status=$(echo "$RESULT" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status","?"))' 2>/dev/null)
        if [ "$status" != "running" ]; then
            echo "$RESULT"
            return 0
        fi
    done
    echo '{"status":"timeout"}'
    return 1
}

count_tool() {
    local name="$1"
    grep -c "PARSED: name=$name " "$TOOLLOG" 2>/dev/null || echo 0
}

MSGFILE=$(mktemp)
trap "rm -f $MSGFILE" EXIT

echo "=== Swift Agent Harness Test ==="
echo "TestServer: $BASE"
echo "Workspace: $WORKSPACE"
echo ""

# ─────────────────────────────────────────────
echo -e "${YELLOW}Phase 1: writeFile (5x)${NC}"
# ─────────────────────────────────────────────
> "$TOOLLOG"
cat > "$MSGFILE" << 'PROMPT'
Create these 5 files using writeFile tool (do NOT use shell). Create each one separately:
- alpha.txt containing: Alpha content line one
- bravo.txt containing: Bravo content line two
- charlie.txt containing: Charlie content line three
- delta.txt containing: Delta content line four
- foxtrot.txt containing: Foxtrot content line five
PROMPT
RESULT=$(run_agent "$MSGFILE" 10 180)

WF_COUNT=$(count_tool "writeFile")
echo -e "  ${DIM}writeFile calls: $WF_COUNT${NC}"
check "writeFile called >= 5 times" "$([ "$WF_COUNT" -ge 5 ] && echo true || echo false)" "got $WF_COUNT"
for f in alpha.txt bravo.txt charlie.txt delta.txt foxtrot.txt; do
    check "File $f exists" "$([ -f "$WORKSPACE/$f" ] && echo true || echo false)"
done

# ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Phase 2: readFile (5x)${NC}"
# ─────────────────────────────────────────────
> "$TOOLLOG"
cat > "$MSGFILE" << 'PROMPT'
Read each of these 5 files using readFile tool and tell me the contents: alpha.txt, bravo.txt, charlie.txt, delta.txt, foxtrot.txt
PROMPT
RESULT=$(run_agent "$MSGFILE" 8 120)

RF_COUNT=$(count_tool "readFile")
echo -e "  ${DIM}readFile calls: $RF_COUNT${NC}"
check "readFile called >= 5 times" "$([ "$RF_COUNT" -ge 5 ] && echo true || echo false)" "got $RF_COUNT"

# ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Phase 3: editFile (5x)${NC}"
# ─────────────────────────────────────────────
> "$TOOLLOG"
cat > "$MSGFILE" << 'PROMPT'
Use editFile to edit each of these 5 files. In each file find the word "line" and replace it with "EDITED". Edit all 5: alpha.txt, bravo.txt, charlie.txt, delta.txt, foxtrot.txt
PROMPT
RESULT=$(run_agent "$MSGFILE" 15 240)

EF_COUNT=$(count_tool "editFile")
echo -e "  ${DIM}editFile calls: $EF_COUNT${NC}"
check "editFile called >= 5 times" "$([ "$EF_COUNT" -ge 5 ] && echo true || echo false)" "got $EF_COUNT"
EDITED=0
for f in alpha.txt bravo.txt charlie.txt delta.txt foxtrot.txt; do
    if [ -f "$WORKSPACE/$f" ] && grep -q 'EDITED' "$WORKSPACE/$f"; then
        EDITED=$((EDITED + 1))
    fi
done
check "Edits applied (>= 3 files)" "$([ "$EDITED" -ge 3 ] && echo true || echo false)" "$EDITED files have EDITED"

# ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Phase 4: shell (5x)${NC}"
# ─────────────────────────────────────────────
> "$TOOLLOG"
cat > "$MSGFILE" << 'PROMPT'
Run these 5 shell commands using the shell tool, one at a time: echo SHELLTEST1, echo SHELLTEST2, echo SHELLTEST3, echo SHELLTEST4, echo SHELLTEST5
PROMPT
RESULT=$(run_agent "$MSGFILE" 8 120)

SH_COUNT=$(count_tool "shell")
echo -e "  ${DIM}shell calls: $SH_COUNT${NC}"
check "shell called >= 5 times" "$([ "$SH_COUNT" -ge 5 ] && echo true || echo false)" "got $SH_COUNT"

# ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Phase 5: searchFiles (5x)${NC}"
# ─────────────────────────────────────────────
> "$TOOLLOG"
cat > "$MSGFILE" << 'PROMPT'
Use the searchFiles tool (not grep, not shell) to search for these 5 patterns one at a time: Alpha, Bravo, Charlie, Delta, Foxtrot
PROMPT
RESULT=$(run_agent "$MSGFILE" 8 120)

SF_COUNT=$(count_tool "searchFiles")
echo -e "  ${DIM}searchFiles calls: $SF_COUNT${NC}"
check "searchFiles called >= 5 times" "$([ "$SF_COUNT" -ge 5 ] && echo true || echo false)" "got $SF_COUNT"

# ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Phase 6: listFiles (5x)${NC}"
# ─────────────────────────────────────────────
> "$TOOLLOG"
mkdir -p "$WORKSPACE/dirA" "$WORKSPACE/dirB" "$WORKSPACE/dirC"
echo "a" > "$WORKSPACE/dirA/a.txt"
echo "b" > "$WORKSPACE/dirB/b.txt"
echo "c" > "$WORKSPACE/dirC/c.txt"

cat > "$MSGFILE" << 'PROMPT'
Use listFiles tool to list these 5 locations: the current directory, dirA, dirB, dirC, and the current directory recursively with recursive true
PROMPT
RESULT=$(run_agent "$MSGFILE" 8 120)

LF_COUNT=$(count_tool "listFiles")
echo -e "  ${DIM}listFiles calls: $LF_COUNT${NC}"
check "listFiles called >= 5 times" "$([ "$LF_COUNT" -ge 5 ] && echo true || echo false)" "got $LF_COUNT"

# ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Phase 7: webSearch (5x)${NC}"
# ─────────────────────────────────────────────
> "$TOOLLOG"
cat > "$MSGFILE" << 'PROMPT'
Use the webSearch tool to search for these 5 queries: Apple MLX, Swift language, Zig language, macOS Sequoia, neural network inference
PROMPT
RESULT=$(run_agent "$MSGFILE" 8 180)

WS_COUNT=$(count_tool "webSearch")
echo -e "  ${DIM}webSearch calls: $WS_COUNT${NC}"
check "webSearch called >= 5 times" "$([ "$WS_COUNT" -ge 5 ] && echo true || echo false)" "got $WS_COUNT"

# ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Phase 8: browse (5x)${NC}"
# ─────────────────────────────────────────────
> "$TOOLLOG"
cat > "$MSGFILE" << 'PROMPT'
Use the browse tool to navigate to these 5 URLs: https://example.com, https://httpbin.org/html, https://httpbin.org/json, https://example.org, https://httpbin.org/robots.txt
PROMPT
RESULT=$(run_agent "$MSGFILE" 8 180)

BR_COUNT=$(count_tool "browse")
echo -e "  ${DIM}browse calls: $BR_COUNT${NC}"
check "browse called >= 5 times" "$([ "$BR_COUNT" -ge 5 ] && echo true || echo false)" "got $BR_COUNT"

# ─────────────────────────────────────────────
echo ""
echo -e "${YELLOW}Phase 9: saveMemory (5x)${NC}"
# ─────────────────────────────────────────────
> "$TOOLLOG"
cat > "$MSGFILE" << 'PROMPT'
Save these 5 memories using saveMemory tool: AGENTTEST_mem1, AGENTTEST_mem2, AGENTTEST_mem3, AGENTTEST_mem4, AGENTTEST_mem5
PROMPT
RESULT=$(run_agent "$MSGFILE" 8 120)

SM_COUNT=$(count_tool "saveMemory")
echo -e "  ${DIM}saveMemory calls: $SM_COUNT${NC}"
check "saveMemory called >= 5 times" "$([ "$SM_COUNT" -ge 5 ] && echo true || echo false)" "got $SM_COUNT"
MEM_COUNT=$(grep -c "AGENTTEST_mem" "$HOME/.mlx-serve/memory.md" 2>/dev/null || echo 0)
check "Memories persisted (>= 3)" "$([ "$MEM_COUNT" -ge 3 ] && echo true || echo false)" "found $MEM_COUNT"

# ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo -e "  ${GREEN}Passed: $PASS${NC}  ${RED}Failed: $FAIL${NC}  Total: $TOTAL"
echo "═══════════════════════════════════════════════"

# Cleanup test files but keep workspace
rm -rf "$WORKSPACE/dirA" "$WORKSPACE/dirB" "$WORKSPACE/dirC"
rm -f "$WORKSPACE"/*.txt "$WORKSPACE"/*.py 2>/dev/null
# Remove test memories
if [ -f "$HOME/.mlx-serve/memory.md" ]; then
    sed -i '' '/AGENTTEST_mem/d' "$HOME/.mlx-serve/memory.md"
fi

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check tool-calls.log for details.${NC}"
    exit 1
fi
