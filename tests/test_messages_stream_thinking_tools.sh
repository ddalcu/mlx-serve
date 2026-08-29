#!/bin/bash
# Regression tests for /v1/messages STREAMING with thinking + tools together —
# the exact Claude Code shape (Claude Code always sends tools, and Qwen 3.5/3.6
# templates inject the `<think>\n` opener into the generation prompt).
#
# Pins the 2026-06-10 live Claude Code failure on Qwen3.6-35B-A3B:
#   1. Template-opened thinking streamed as visible text_delta events,
#      leaking a raw `</think>` token into the transcript (the tools branch
#      only recognized think OPENERS present in the model's output).
#   2. On a tool-call turn, the end-of-stream emission opened a thinking
#      block at an index already occupied by the open text block, then
#      closed the text block at a never-started index — Claude Code aborted
#      the turn with "API Error: Content block not found".
#
# Checks (per prompt): no `</think>`/`<think>` in any text_delta, thinking
# arrives as thinking_delta events, and the content-block lifecycle is
# protocol-valid (every delta/stop references an open block, no index reuse
# while open, nothing left open at message_stop).
#
# Usage: ./tests/test_messages_stream_thinking_tools.sh [model_dir] [port]
#   model_dir should be a template-opened-think model (Qwen 3.5/3.6).
#   Default: the model family from the live failure. Starts its own server.

set -u

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/Qwen3.6-35B-A3B-6bit}"
PORT="${2:-11263}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL"
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 1
"$BINARY" --model "$MODEL" --serve --port "$PORT" --ctx-size 8192 --log-level info > /tmp/test_messages_stream_thinking_tools.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "FAIL: server did not come up"; exit 1; }

TOOLS='[{"name":"get_weather","description":"Get current weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'

# Validate an SSE capture: block lifecycle + no think tags in text deltas.
# Prints "OK <n_text> <n_thinking> <n_tool_use>" or "ERR <reason>".
validate() {
    python3 - "$1" <<'EOF'
import json, sys

open_blocks = {}   # index -> type
counts = {"text": 0, "thinking": 0, "tool_use": 0}
err = None
saw_message_stop = False

for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data:"):
        continue
    try:
        ev = json.loads(line[5:].strip())
    except json.JSONDecodeError:
        err = err or "unparseable SSE data line"
        continue
    t = ev.get("type")
    if t == "content_block_start":
        idx = ev["index"]
        btype = ev["content_block"]["type"]
        if idx in open_blocks:
            err = err or f"start index {idx} while already open as {open_blocks[idx]}"
        open_blocks[idx] = btype
        counts[btype] = counts.get(btype, 0) + 1
    elif t == "content_block_delta":
        idx = ev["index"]
        if idx not in open_blocks:
            err = err or f"delta for unopened index {idx}"
        d = ev.get("delta", {})
        if d.get("type") == "text_delta":
            txt = d.get("text", "")
            if "</think>" in txt or "<think>" in txt:
                err = err or f"think tag leaked in text_delta: {txt!r}"
    elif t == "content_block_stop":
        idx = ev["index"]
        if idx not in open_blocks:
            err = err or f"stop for unopened index {idx}"
        else:
            del open_blocks[idx]
    elif t == "message_stop":
        saw_message_stop = True
        if open_blocks:
            err = err or f"blocks still open at message_stop: {sorted(open_blocks)}"

if not saw_message_stop:
    err = err or "no message_stop event"
if err:
    print(f"ERR {err}")
else:
    print(f"OK {counts['text']} {counts['thinking']} {counts['tool_use']}")
EOF
}

run_stream() { # body -> capture file
    local body="$1" out="$2"
    curl -sN "$BASE/v1/messages" -H 'Content-Type: application/json' -d "$body" > "$out"
}

echo "1. thinking+tools stream, plain answer (no tool expected)"
BODY1=$(cat <<EOF
{"model":"m","max_tokens":400,"stream":true,
 "thinking":{"type":"enabled","budget_tokens":8000},
 "tools":$TOOLS,
 "messages":[{"role":"user","content":"What is 17+25? Answer with just the number. Do not use any tools."}]}
EOF
)
run_stream "$BODY1" /tmp/msgs_stream_tt_1.sse
V1=$(validate /tmp/msgs_stream_tt_1.sse)
echo "    -> $V1"
check "protocol-valid block lifecycle, no think-tag leak" "$([ "${V1%% *}" = "OK" ] && echo 1 || echo 0)"
N_THINK1=$(echo "$V1" | awk '{print $3}')
check "thinking arrives as thinking_delta block(s)" "$([ "${N_THINK1:-0}" -ge 1 ] 2>/dev/null && echo 1 || echo 0)"
N_TEXT1=$(echo "$V1" | awk '{print $2}')
check "visible answer arrives as text block(s)" "$([ "${N_TEXT1:-0}" -ge 1 ] 2>/dev/null && echo 1 || echo 0)"

echo "2. thinking+tools stream, tool-call turn (the Claude Code failure)"
BODY2=$(cat <<EOF
{"model":"m","max_tokens":600,"stream":true,
 "thinking":{"type":"enabled","budget_tokens":8000},
 "tools":$TOOLS,
 "messages":[{"role":"user","content":"What is the weather in Paris right now? Use the get_weather tool."}]}
EOF
)
run_stream "$BODY2" /tmp/msgs_stream_tt_2.sse
V2=$(validate /tmp/msgs_stream_tt_2.sse)
echo "    -> $V2"
check "protocol-valid block lifecycle, no think-tag leak" "$([ "${V2%% *}" = "OK" ] && echo 1 || echo 0)"
N_TOOL2=$(echo "$V2" | awk '{print $4}')
check "tool_use block emitted" "$([ "${N_TOOL2:-0}" -ge 1 ] 2>/dev/null && echo 1 || echo 0)"
grep -q '"stop_reason":"tool_use"' /tmp/msgs_stream_tt_2.sse
check "stop_reason is tool_use" "$([ $? -eq 0 ] && echo 1 || echo 0)"

echo ""
echo "===== $PASS passed, $FAIL failed ====="
[ "$FAIL" -eq 0 ] || exit 1
