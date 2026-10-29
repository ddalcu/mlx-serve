#!/bin/bash
# test_multi_image_history.sh — every image in a conversation reaches the model where the template put it: turn 1
# image A, turn 2 image B, turn 3 asks about A (the answer needs A's pixels, the log shows `Multimodal: 2 item(s)`);
# a repeat of turn 3 is warm across both media spans with the same bytes; an image inside an Anthropic tool_result
# (Claude Code's Read) and one in a Responses function_call_output (Codex's view_image) are described.
#
#   ./tests/test_multi_image_history.sh [model] [port]   # default Qwen3.5-2B
#
# Hermetic twins: expandMediaPlaceholders, `media renders one template placeholder per item`.

set -u
MODEL="${MULTI_IMAGE_MODEL:-${1:-$HOME/.mlx-serve/models/lmstudio-community/Qwen3.5-2B-MLX-4bit}}"
PORT="${2:-11236}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
LOG="$HOME/claude-tmp/multi-image/server-$PORT.log"
mkdir -p "$(dirname "$LOG")"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no model at $MODEL"; exit 0; }
A="tests/fixtures/street-name-signs.jpg"
B="tests/fixtures/house.jpeg"
for f in "$A" "$B"; do [ -f "$f" ] || { echo "SKIP: fixture $f missing"; exit 0; }; done
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
has() { echo "$1" | grep -ciE "$2" | sed 's/^[1-9][0-9]*$/1/'; }

"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level debug > "$LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
U="http://127.0.0.1:$PORT"
for _ in $(seq 1 300); do curl -sf "$U/health" >/dev/null 2>&1 && grep -q "ready" "$LOG" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$LOG"; exit 1; }; sleep 2; done

openai_body() { # $1 turns (2 or 3)
  python3 - "$A" "$B" "$1" <<'PY'
import base64, json, sys
a, b, turns = sys.argv[1], sys.argv[2], int(sys.argv[3])
url = lambda p: "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()
msgs = [
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(a)}}, {"type": "text", "text": "Here is the first picture."}]},
    {"role": "assistant", "content": "Got it."},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(b)}}, {"type": "text", "text": "Here is the second picture."}]},
]
if turns == 3:
    msgs += [{"role": "assistant", "content": "Got it."},
             {"role": "user", "content": "What text is written on the green street signs in the FIRST picture? Answer with the words only."}]
print(json.dumps({"model": "mlx-serve", "max_tokens": 48, "temperature": 0, "enable_thinking": False, "messages": msgs}))
PY
}
ask() { curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens_details']['cached_tokens'], '|', d['choices'][0]['message']['content'].replace(chr(10),' '))"; }

echo "[1] turn 3 asks about the turn-1 image"
r1=$(openai_body 3 | ask); echo "  $r1"
check "both images encoded" "$(grep -c 'Multimodal: 2 item' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "answer reads image A" "$(has "$r1" 'gr[ae]y fox|waterfall')" "1"

echo "[2] the same turn again: warm across both spans, same bytes"
r2=$(openai_body 3 | ask); echo "  $r2"
check "warm: cached_tokens > 0" "$(python3 -c "print(1 if int('${r2%% |*}')>0 else 0)")" "1"
check "warm == cold bytes" "${r2#* | }" "${r1#* | }"

echo "[3] Anthropic tool_result image reaches the model"
r3=$(python3 - "$B" <<'PY' | curl -s -m 600 "$U/v1/messages" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(''.join(x.get('text','') for x in d.get('content',[]) if x.get('type')=='text').replace(chr(10),' '))"
import base64, json, sys
data = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
print(json.dumps({
    "model": "mlx-serve", "max_tokens": 1024, "temperature": 0,
    "tools": [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
    "messages": [
        {"role": "user", "content": "Open photo.jpg and tell me what it shows."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "photo.jpg"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}]},
            {"type": "text", "text": "What is the main subject of that photo? One short sentence, no tool calls."},
        ]},
    ],
}))
PY
)
echo "  $r3"
check "tool_result image encoded" "$(grep -c 'Multimodal: 1 item' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "answer describes the house" "$(has "$r3" 'house|home|building')" "1"

echo "[4] Responses function_call_output image reaches the model"
n4=$(grep -c 'Multimodal: 1 item' "$LOG")
r4=$(python3 - "$B" <<'PY' | curl -s -m 600 "$U/v1/responses" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(''.join(c.get('text','') for o in d.get('output',[]) if o.get('type')=='message' for c in o.get('content',[])).replace(chr(10),' '))"
import base64, json, sys
data = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
print(json.dumps({
    "model": "mlx-serve", "max_output_tokens": 1024, "temperature": 0, "tool_choice": "none",
    "tools": [{"type": "function", "name": "view_image", "description": "View an image file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
    "input": [
        {"role": "user", "content": "Open photo.jpg and tell me what it shows."},
        {"type": "function_call", "call_id": "c1", "name": "view_image", "arguments": "{\"path\":\"photo.jpg\"}"},
        {"type": "function_call_output", "call_id": "c1", "output": [
            {"type": "input_text", "text": "Read photo.jpg"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + data},
        ]},
        {"role": "user", "content": "What is the main subject of that photo? One short sentence, no tool calls."},
    ],
}))
PY
)
echo "  $r4"
check "function_call_output image encoded" "$(( $(grep -c 'Multimodal: 1 item' "$LOG") - n4 ))" "1"
check "function_call_output answer describes the house" "$(has "$r4" 'house|home|building')" "1"

echo "pass=$pass fail=$fail"
[ "$fail" = "0" ] && echo "PASS: multi-image history" || { echo "FAIL: multi-image history"; grep -E "Multimodal|hot-cache|cache\]|-> 4|-> 5" "$LOG" | tail -20; }
[ "$fail" = "0" ]
