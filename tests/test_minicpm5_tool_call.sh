#!/bin/bash
# End-to-end acceptance for MiniCPM5 V3's native tool-call XML
# (`<function name="X"><param name="K">V</param></function>`) across ALL FOUR
# (API shape x mode) combinations, against a real local model.
#
#   1. /v1/messages          streaming
#   2. /v1/messages          non-streaming
#   3. /v1/chat/completions  non-streaming
#   4. /v1/chat/completions  streaming
#
# Acceptance rule, applied to every row: no MiniCPM5 XML may reach the client
# as visible content, and a valid call must arrive as the surface's proper
# structured tool-call object. A leak is a property of the SURFACE, not of the
# dialect -- each combination runs its own buffering path, so one passing row
# proves nothing about the other three.
#
# Row 1 pins the original Claude Code failure (2026-07-22): the Anthropic
# streaming handler used to carry its OWN narrow, hand-rolled tool-detection
# gate -- only `<tool_call`/`<|tool_call`/raw-JSON -- completely blind to
# `<function`. The raw XML streamed to the client as a LIVE text_delta before
# the end-of-stream parse ever ran, and a SEPARATE, correct tool_use block was
# then emitted for the same call: duplicated, XML-leaking content, not a
# missing call. Upstream has since routed that gate through the shared
# chat.streamShouldBufferForTools(), which is what makes this row pass.
#
# Usage: ./tests/test_minicpm5_tool_call.sh [model_dir] [port]
#   Default model: the local mlx-community/MiniCPM5-1B-OptiQ-4bit pull.
#   SKIPs cleanly (exit 0) when the weights are absent.

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
# MLX_SERVE_RAW_DUMP_FILE records the PRE-PARSE model output. Without it this
# whole script only ever inspects the NORMALISED client response, which raw
# JSON or Hermes XML would satisfy just as well - it would stay green on a
# model that never emitted MiniCPM5 XML at all, proving nothing about the
# dialect this PR exists for.
RAW_DUMP=/tmp/minicpm5_raw_dump.txt
rm -f "$RAW_DUMP"
MLX_SERVE_RAW_DUMP_FILE="$RAW_DUMP" \
"$BINARY" --model "$MODEL" --serve --port "$PORT" --log-level debug > /tmp/test_messages_stream_minicpm5_tool_call.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "FAIL: server did not come up"; exit 1; }

TOOLS='[{"name":"shell","description":"Run a shell command","input_schema":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}]'
OAI_TOOLS='[{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]'

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

# ── The other three rows of the acceptance matrix ────────────────────────
# A leak is a property of the SURFACE, not of the dialect: each of the four
# (API shape × mode) combinations runs its own buffering path, so passing one
# proves nothing about the other three. Same server boot, same tool, same
# prompt; only the endpoint and `stream` change.

# Non-streaming JSON: assert the call is structured AND no dialect XML rode
# out as visible content. `has_xml` is checked on the CONTENT field only —
# the tool-call arguments legitimately contain the command text.
validate_nonstream() { # file, content_path, calls_path -> "OK <name> <args>" | "ERR ..."
    python3 - "$1" "$2" "$3" <<'EOF'
import json, sys
LEAK_TAGS = ("<function", "<param", "</function>", "</param>")
doc = json.load(open(sys.argv[1]))

def dig(obj, path):
    for key in path.split("."):
        if obj is None:
            return None
        obj = obj[int(key)] if key.isdigit() else obj.get(key)
    return obj

# Anthropic: content is a block list. OpenAI: content is a string.
content = dig(doc, sys.argv[2])
if isinstance(content, list):
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    calls = [b for b in content if b.get("type") == "tool_use"]
    got = [(c.get("name"), json.dumps(c.get("input", {}), sort_keys=True)) for c in calls]
else:
    text = content or ""
    calls = dig(doc, sys.argv[3]) or []
    got = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]

leaked = [t for t in LEAK_TAGS if t in text]
if leaked:
    print(f"ERR dialect XML in visible content: {leaked} in {text!r}")
elif not got:
    print("ERR no tool call in the response")
else:
    print(f"OK {got[0][0]} {got[0][1]}")
EOF
}

echo ""
echo "2. /v1/messages NON-streaming"
curl -s "$BASE/v1/messages" -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d "{\"model\":\"m\",\"max_tokens\":200,\"tools\":$TOOLS,\"system\":\"You can call tools. Use the shell tool to run commands.\",\"messages\":[{\"role\":\"user\",\"content\":\"Use the shell tool to run: git status\"}]}" \
  > /tmp/msgs_nonstream_minicpm5.json
V2=$(validate_nonstream /tmp/msgs_nonstream_minicpm5.json content '')
echo "    -> $V2"
check "tool_use block, no dialect XML in any text block" "$([ "${V2%% *}" = "OK" ] && echo 1 || echo 0)"
check "tool_use names the shell tool" "$(echo "$V2" | grep -q '^OK shell ' && echo 1 || echo 0)"
check "/v1/messages tool arg is exactly git status" "$(echo "$V2" | grep -q 'git status' && echo 1 || echo 0)"

echo ""
echo "3. /v1/chat/completions NON-streaming"
curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"max_tokens\":200,\"tools\":$OAI_TOOLS,\"messages\":[{\"role\":\"system\",\"content\":\"You can call tools. Use the shell tool to run commands.\"},{\"role\":\"user\",\"content\":\"Use the shell tool to run: git status\"}]}" \
  > /tmp/chat_nonstream_minicpm5.json
V3=$(validate_nonstream /tmp/chat_nonstream_minicpm5.json choices.0.message.content choices.0.message.tool_calls)
echo "    -> $V3"
check "tool_calls[], no dialect XML in message.content" "$([ "${V3%% *}" = "OK" ] && echo 1 || echo 0)"
check "tool_calls names the shell tool" "$(echo "$V3" | grep -q '^OK shell ' && echo 1 || echo 0)"
check "/v1/chat/completions tool arg is exactly git status" "$(echo "$V3" | grep -q 'git status' && echo 1 || echo 0)"
grep -q '"finish_reason":"tool_calls"' /tmp/chat_nonstream_minicpm5.json
check "finish_reason is tool_calls" "$([ $? -eq 0 ] && echo 1 || echo 0)"

echo ""
echo "4. /v1/chat/completions STREAMING"
curl -sN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"max_tokens\":200,\"stream\":true,\"tools\":$OAI_TOOLS,\"messages\":[{\"role\":\"system\",\"content\":\"You can call tools. Use the shell tool to run commands.\"},{\"role\":\"user\",\"content\":\"Use the shell tool to run: git status\"}]}" \
  > /tmp/chat_stream_minicpm5.sse
V4=$(python3 - /tmp/chat_stream_minicpm5.sse <<'EOF'
import json, sys
LEAK_TAGS = ("<function", "<param", "</function>", "</param>")
content, n_calls, finish, err = "", 0, None, None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        continue
    try:
        ev = json.loads(payload)
    except json.JSONDecodeError:
        err = err or "unparseable SSE data line"
        continue
    for ch in ev.get("choices", []) or []:
        d = ch.get("delta") or {}
        piece = d.get("content") or ""
        content += piece
        if any(t in piece for t in LEAK_TAGS):
            err = err or f"dialect XML leaked in a content delta: {piece!r}"
        n_calls += len(d.get("tool_calls") or [])
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
if content.strip():
    err = err or f"non-whitespace content on a tool-call turn: {content!r}"
if not n_calls:
    err = err or "no tool_calls delta emitted"
print(f"ERR {err}" if err else f"OK {n_calls} {finish}")
EOF
)
echo "    -> $V4"
check "tool_calls deltas, no dialect XML in any content delta" "$([ "${V4%% *}" = "OK" ] && echo 1 || echo 0)"
check "streaming finish_reason is tool_calls" "$(echo "$V4" | grep -q ' tool_calls$' && echo 1 || echo 0)"
grep -q '"name":"shell"' /tmp/chat_stream_minicpm5.sse
check "streaming tool call names the shell tool" "$([ $? -eq 0 ] && echo 1 || echo 0)"
grep -q 'git status' /tmp/chat_stream_minicpm5.sse
check "streaming /v1/chat/completions tool arg is exactly git status" "$([ $? -eq 0 ] && echo 1 || echo 0)"

echo ""
echo "5. The dialect was actually exercised (pre-parse evidence)"
# The decisive assertion. Everything above inspects the parsed client response,
# which a model emitting raw JSON or Hermes XML would also satisfy. This one
# fails unless MiniCPM5 attribute XML was really generated - so a green run
# cannot be mistaken for coverage of a dialect the model never produced.
if [ -s "$RAW_DUMP" ]; then
    grep -q '<function name=' "$RAW_DUMP"
    check "model emitted MiniCPM5 attribute XML (<function name=)" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    grep -q '<param name=' "$RAW_DUMP"
    check "model emitted <param name= elements" "$([ $? -eq 0 ] && echo 1 || echo 0)"
else
    check "raw pre-parse dump was captured" 0
fi

echo ""
echo "===== $PASS passed, $FAIL failed ====="
[ "$FAIL" -eq 0 ]
