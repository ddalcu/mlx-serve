#!/bin/bash
# PLD + tools equivalence test.
#
# Agent clients (Claude Code, the app's agent loop) send `tools` on every
# request, and agent traffic is the ideal PLD workload — tool results echo
# file contents the model then quotes back. Historically PLD was blanket-
# disabled whenever tools were present ("pld=disabled (tools present)"), a
# conservative gate from the original PLD checkpoint that predates streaming
# PLD, scheduler slots, and the runtime acceptance gate.
#
# This test asserts:
#   1. a tools request with enable_pld:true actually runs PLD
#      (log shows "[spec-stats] mode=pld", not "pld=disabled (tools present)")
#   2. the tool call (name + args) is identical to a no-PLD baseline
#   3. an echo-heavy text answer with tools defined is byte-identical
#   4. a streaming /v1/messages tools request still parses the tool call
#
# Requires:
#   - A built mlx-serve binary (zig build -Doptimize=ReleaseFast)
#   - PLD_TEST_MODEL or ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit
#
# Usage: ./tests/test_pld_tools.sh [port]

set -e

PORT=${1:-8098}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${PLD_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_pld_tools: model directory not found."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

FAILURES=0
check_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo -e "  ${GREEN}PASS${NC} $desc"
    else
        echo -e "  ${RED}FAIL${NC} $desc"
        echo "    expected: $expected"
        echo "    actual:   $actual"
        FAILURES=$((FAILURES + 1))
    fi
}
check_grep() {
    local desc="$1" pattern="$2" file="$3"
    if grep -q "$pattern" "$file"; then
        echo -e "  ${GREEN}PASS${NC} $desc"
    else
        echo -e "  ${RED}FAIL${NC} $desc (no '$pattern' in server log)"
        FAILURES=$((FAILURES + 1))
    fi
}

TOOLS_CC='[{"type":"function","function":{"name":"write_file","description":"Write content to a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]'

# Echo-heavy tool prompt: the args the model must produce appear verbatim in
# the prompt, so PLD's n-gram lookup gets real acceptance.
TOOL_PROMPT="Call write_file with path src/hello.zig and content: pub fn main() !void { std.debug.print(\\\"hello\\\", .{}); } Use exactly that path and content."
ECHO_PROMPT="Repeat the following line exactly, three times, nothing else: const value = compute_total(alpha, beta, gamma);"

start_server() {
    local logfile="$1"; shift
    "$BINARY" --model "$MODEL" --serve --port "$PORT" "$@" > "$logfile" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 60); do
        curl -s -f "$BASE/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    echo -e "${RED}FAIL${NC} server did not become healthy"; tail -5 "$logfile"; return 1
}
stop_server() { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; }
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

tool_request() {
    local extra="$1"
    curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\":\"mlx-serve\",\"temperature\":0,\"max_tokens\":128,$extra
      \"tools\":$TOOLS_CC,
      \"messages\":[{\"role\":\"user\",\"content\":\"$TOOL_PROMPT\"}]}"
}
extract_call() {
    python3 -c "
import sys, json
r = json.load(sys.stdin)
tc = (r['choices'][0]['message'].get('tool_calls') or [None])[0]
if tc is None:
    print('NO_TOOL_CALL'); sys.exit(0)
args = json.loads(tc['function']['arguments'])
print(tc['function']['name'] + '|' + json.dumps(args, sort_keys=True))"
}
echo_request() {
    local extra="$1"
    curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\":\"mlx-serve\",\"temperature\":0,\"max_tokens\":96,$extra
      \"tools\":$TOOLS_CC,
      \"messages\":[{\"role\":\"user\",\"content\":\"$ECHO_PROMPT\"}]}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content') or '')"
}

echo "1) Baseline server (no PLD)..."
LOG_A=$(mktemp)
start_server "$LOG_A"
BASE_CALL=$(tool_request "" | extract_call)
BASE_ECHO=$(echo_request "")
stop_server
echo "   baseline call: $BASE_CALL"
if [ "$BASE_CALL" = "NO_TOOL_CALL" ]; then
    echo -e "${RED}FAIL${NC} baseline produced no tool call; prompt needs adjusting"; exit 1
fi

echo "2) PLD server (--pld), tools request with enable_pld:true..."
LOG_B=$(mktemp)
start_server "$LOG_B" --pld
PLD_CALL=$(tool_request "\"enable_pld\":true," | extract_call)
PLD_ECHO=$(echo_request "\"enable_pld\":true,")

# Streaming /v1/messages (the Claude Code shape) with tools + PLD.
STREAM_TOOL=$(curl -s -N -m 120 "$BASE/v1/messages" -H 'Content-Type: application/json' -d "{
  \"model\":\"mlx-serve\",\"temperature\":0,\"max_tokens\":128,\"stream\":true,\"enable_pld\":true,
  \"tools\":[{\"name\":\"write_file\",\"description\":\"Write content to a file.\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}],
  \"messages\":[{\"role\":\"user\",\"content\":\"$TOOL_PROMPT\"}]}" \
| python3 -c "
import sys, json
name, parts = None, []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: '): continue
    try: ev = json.loads(line[6:])
    except Exception: continue
    if ev.get('type') == 'content_block_start' and ev['content_block'].get('type') == 'tool_use':
        name = ev['content_block']['name']
    if ev.get('type') == 'content_block_delta' and ev['delta'].get('type') == 'input_json_delta':
        parts.append(ev['delta']['partial_json'])
if name is None:
    print('NO_TOOL_CALL')
else:
    print(name + '|' + json.dumps(json.loads(''.join(parts)), sort_keys=True))")
stop_server

check_grep "PLD actually engaged on tools request" "\[spec-stats\] mode=pld" "$LOG_B"
if grep -q "pld=disabled (tools present)" "$LOG_B"; then
    echo -e "  ${RED}FAIL${NC} server still blanket-disables PLD when tools are present"
    FAILURES=$((FAILURES + 1))
else
    echo -e "  ${GREEN}PASS${NC} no 'pld=disabled (tools present)' gate"
fi
check_eq "tool call identical under PLD" "$BASE_CALL" "$PLD_CALL"
check_eq "echo text identical under PLD" "$BASE_ECHO" "$PLD_ECHO"
check_eq "streaming /v1/messages tool call identical under PLD" "$BASE_CALL" "$STREAM_TOOL"

rm -f "$LOG_A" "$LOG_B"
if [ "$FAILURES" -gt 0 ]; then
    echo -e "${RED}FAIL${NC} $FAILURES check(s) failed"
    exit 1
fi
echo -e "${GREEN}PASS${NC} PLD + tools: engaged, byte-identical, tool calls intact"
