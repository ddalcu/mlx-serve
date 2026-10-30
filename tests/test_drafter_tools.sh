#!/bin/bash
# test_drafter_tools.sh — the Gemma 4 assistant drafter with tools present, test_pld_tools.sh's contract: chat
# non-stream, /v1/messages stream + non-stream and /v1/responses non-stream each count a `[spec-stats] mode=drafter`
# engagement (the regular-decode fallback is output-identical, so equality alone misses a dispatch hole) and match a
# no-drafter baseline byte for byte.
#
#   DRAFTER_TEST_TARGET=<gemma-4 dense dir> DRAFTER_TEST_DRAFTER=<assistant dir> ./tests/test_drafter_tools.sh [port]

set -e
source "$(dirname "$0")/lib/server.sh"

PORT=${1:-8097}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${DRAFTER_TEST_TARGET:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
DRAFTER="${DRAFTER_TEST_DRAFTER:-$HOME/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16}"
if [ ! -d "$MODEL" ] || [ ! -d "$DRAFTER" ]; then
    echo -e "${YELLOW}SKIP${NC} test_drafter_tools: target or drafter checkpoint not found."
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
TOOL_PROMPT="Call write_file with path src/hello.zig and content: pub fn main() !void { std.debug.print(\\\"hello\\\", .{}); } Use exactly that path and content."
ECHO_PROMPT="Repeat the following line exactly, three times, nothing else: const value = compute_total(alpha, beta, gamma);"

start_server() {
    local logfile="$1"; shift
    "$BINARY" --model "$MODEL" --serve --port "$PORT" "$@" > "$logfile" 2>&1 &
    SERVER_PID=$!
    srv_wait_health "$BASE" 90 1 && return 0
    echo -e "${RED}FAIL${NC} server did not become healthy"; tail -5 "$logfile"; return 1
}
stop_server() { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; }
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

tool_request() {
    local extra="$1"
    curl -s -m 180 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
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
    curl -s -m 180 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
      \"model\":\"mlx-serve\",\"temperature\":0,\"max_tokens\":96,$extra
      \"tools\":$TOOLS_CC,
      \"messages\":[{\"role\":\"user\",\"content\":\"$ECHO_PROMPT\"}]}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message'].get('content') or '')"
}

echo "1) Baseline server (no drafter)..."
LOG_A=$(mktemp)
start_server "$LOG_A"
BASE_CALL=$(tool_request "" | extract_call)
BASE_ECHO=$(echo_request "")
stop_server
echo "   baseline call: $BASE_CALL"
if [ "$BASE_CALL" = "NO_TOOL_CALL" ]; then
    echo -e "${RED}FAIL${NC} baseline produced no tool call; prompt needs adjusting"; exit 1
fi

echo "2) Drafter server (--drafter), tools request with enable_drafter:true..."
LOG_B=$(mktemp)
start_server "$LOG_B" --drafter "$DRAFTER"
DRAFT_CALL=$(tool_request "\"enable_drafter\":true," | extract_call)
DRAFT_ECHO=$(echo_request "\"enable_drafter\":true,")

STREAM_TOOL=$(curl -s -N -m 180 "$BASE/v1/messages" -H 'Content-Type: application/json' -d "{
  \"model\":\"mlx-serve\",\"temperature\":0,\"max_tokens\":128,\"stream\":true,\"enable_drafter\":true,
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
# Non-streaming /v1/messages with tools + drafter (was a hardcoded
# use_drafter=false at the nonStreamingViaScheduler call site).
NS_MSG=$(curl -s -m 180 "$BASE/v1/messages" -H 'Content-Type: application/json' -d "{
  \"model\":\"mlx-serve\",\"temperature\":0,\"max_tokens\":128,\"enable_drafter\":true,
  \"tools\":[{\"name\":\"write_file\",\"description\":\"Write content to a file.\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}],
  \"messages\":[{\"role\":\"user\",\"content\":\"$TOOL_PROMPT\"}]}" \
| python3 -c "
import sys, json
r = json.load(sys.stdin)
tu = next((b for b in r.get('content', []) if b.get('type') == 'tool_use'), None)
if tu is None:
    print('NO_TOOL_CALL')
else:
    print(tu['name'] + '|' + json.dumps(tu['input'], sort_keys=True))")

# Non-streaming /v1/responses with the same tool ask (flat tool form).
NS_RESP=$(curl -s -m 180 "$BASE/v1/responses" -H 'Content-Type: application/json' -d "{
  \"model\":\"mlx-serve\",\"temperature\":0,\"max_output_tokens\":128,\"enable_drafter\":true,
  \"tools\":[{\"type\":\"function\",\"name\":\"write_file\",\"description\":\"Write content to a file.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}],
  \"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"$TOOL_PROMPT\"}]}]}" \
| python3 -c "
import sys, json
r = json.load(sys.stdin)
fc = next((it for it in r.get('output', []) if it.get('type') == 'function_call'), None)
if fc is None:
    print('NO_TOOL_CALL')
else:
    print(fc['name'] + '|' + json.dumps(json.loads(fc['arguments']), sort_keys=True))")
stop_server

check_grep "drafter actually engaged on tools request" "\[spec-stats\] mode=drafter" "$LOG_B"
# Per-request engagement: 5 drafter-driven generations hit this server
# (chat non-stream tool, chat non-stream echo, messages stream tool,
# messages non-stream tool, responses non-stream tool). A silent fallback
# to regular decode produces identical output, so equality checks alone
# can't catch a dispatch hole — count the spec-stats lines.
DRAFTER_RUNS=$(grep -c "\[spec-stats\] mode=drafter" "$LOG_B")
if [ "$DRAFTER_RUNS" -ge 5 ]; then
    echo -e "  ${GREEN}PASS${NC} drafter engaged on all 5 requests (spec-stats count=$DRAFTER_RUNS)"
else
    echo -e "  ${RED}FAIL${NC} drafter engaged on only $DRAFTER_RUNS/5 requests"
    FAILURES=$((FAILURES + 1))
fi
check_eq "non-stream /v1/messages tool call identical under drafter" "$BASE_CALL" "$NS_MSG"
check_eq "non-stream /v1/responses tool call identical under drafter" "$BASE_CALL" "$NS_RESP"
if grep -q "drafter=disabled (tools present)" "$LOG_B"; then
    echo -e "  ${RED}FAIL${NC} server still blanket-disables the drafter when tools are present"
    FAILURES=$((FAILURES + 1))
else
    echo -e "  ${GREEN}PASS${NC} no 'drafter=disabled (tools present)' gate"
fi
check_eq "tool call identical under drafter" "$BASE_CALL" "$DRAFT_CALL"
check_eq "echo text identical under drafter" "$BASE_ECHO" "$DRAFT_ECHO"
check_eq "streaming /v1/messages tool call identical under drafter" "$BASE_CALL" "$STREAM_TOOL"

rm -f "$LOG_A" "$LOG_B"
if [ "$FAILURES" -gt 0 ]; then
    echo -e "${RED}FAIL${NC} $FAILURES check(s) failed"
    exit 1
fi
echo -e "${GREEN}PASS${NC} drafter + tools: engaged, byte-identical, tool calls intact"
