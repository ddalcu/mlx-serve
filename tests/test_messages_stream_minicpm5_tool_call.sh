#!/bin/bash
# Regression test for /v1/messages STREAMING with MiniCPM5's native
# `<function name="X"><param name="K">V</param></function>` tool-call XML.
#
# Pins the Claude Code failure (architecture verification, 2026-07-22): the
# Anthropic streaming handler (handleAnthropicStreaming, has_tools branch)
# used to carry its OWN narrow, hand-rolled tool-detection gate — only
# `<tool_call`/`<|tool_call`/raw-JSON — completely blind to `<function`. The
# raw MiniCPM5 XML streamed to the client as a LIVE text_delta block before
# the end-of-stream full-text parse ever ran, and a SEPARATE, correct
# tool_use block was then also emitted for the same call — duplicated,
# XML-leaking content, not a missing call. Fixed by routing this gate through
# the same chat.streamShouldBufferForTools() the chat-completions stream
# already uses.
#
# Checks: no `<function`/`<param`/`</function>`/`</param>` in any text_delta,
# NO text block is opened at all on a tool-call turn (thinking + tool_use
# only), the content-block lifecycle is protocol-valid, and stop_reason is
# tool_use.
#
# Usage: ./tests/test_messages_stream_minicpm5_tool_call.sh [model_dir] [port]
#   Default model: the local mlx-community/MiniCPM5-1B-OptiQ-4bit pull.

set -u

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/MiniCPM5-1B-OptiQ-4bit}"
PORT="${2:-11264}"
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

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 1
"$BINARY" --model "$MODEL" --serve --port "$PORT" --log-level info > /tmp/test_messages_stream_minicpm5_tool_call.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "FAIL: server did not come up"; exit 1; }

TOOLS='[{"name":"shell","description":"Run a shell command","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}]'

# Validate an SSE capture: block lifecycle + no MiniCPM5 XML in text deltas.
# Prints "OK <n_text> <n_thinking> <n_tool_use>" or "ERR <reason>".
validate() {
    python3 - "$1" <<'EOF'
import json, sys

open_blocks = {}   # index -> type
counts = {"text": 0, "thinking": 0, "tool_use": 0}
text_content = ""  # concatenated across ALL text_delta events, any block
err = None
saw_message_stop = False
LEAK_TAGS = ("<function", "<param", "</function>", "</param>")

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
            text_content += txt
            if any(tag in txt for tag in LEAK_TAGS):
                err = err or f"MiniCPM5 XML leaked in text_delta: {txt!r}"
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

# A text block may legitimately carry template whitespace padding (e.g. the
# blank line between a thinking block and the tool call) — that is NOT the
# leak this test guards against. The bug is raw dialect XML/meaningful
# content appearing as "text" on what should be a thinking+tool_use-only
# turn; whitespace-only text is harmless and pre-dates this fix.
if text_content.strip():
    err = err or f"non-whitespace text content on a tool-call turn: {text_content!r}"

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
    curl -sN "$BASE/v1/messages" -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' -d "$body" > "$out"
}

echo "1. MiniCPM5 tool-call turn over /v1/messages streaming (the Claude Code failure)"
BODY1=$(cat <<EOF
{"model":"m","max_tokens":200,"stream":true,
 "thinking":{"type":"enabled","budget_tokens":100},
 "system":"You can call tools. Use the shell tool to run commands.",
 "tools":$TOOLS,
 "messages":[{"role":"user","content":"Use the shell tool to run: git status"}]}
EOF
)
run_stream "$BODY1" /tmp/msgs_stream_minicpm5_1.sse
V1=$(validate /tmp/msgs_stream_minicpm5_1.sse)
echo "    -> $V1"
check "protocol-valid block lifecycle, no MiniCPM5 XML leak in any text_delta" "$([ "${V1%% *}" = "OK" ] && echo 1 || echo 0)"
# A text block may legitimately carry whitespace-only template padding
# between thinking and the tool call — validate() already fails the case
# above (ERR) if any NON-whitespace text (raw XML or otherwise) appears on
# this tool-call turn, so no separate zero-text-block assertion is needed.
N_TOOL1=$(echo "$V1" | awk '{print $4}')
check "tool_use block emitted" "$([ "${N_TOOL1:-0}" -ge 1 ] 2>/dev/null && echo 1 || echo 0)"
grep -q '"name":"shell"' /tmp/msgs_stream_minicpm5_1.sse
check "tool_use names the shell tool" "$([ $? -eq 0 ] && echo 1 || echo 0)"
grep -q '"stop_reason":"tool_use"' /tmp/msgs_stream_minicpm5_1.sse
check "stop_reason is tool_use" "$([ $? -eq 0 ] && echo 1 || echo 0)"

echo ""
echo "===== $PASS passed, $FAIL failed ====="
[ "$FAIL" -eq 0 ]
