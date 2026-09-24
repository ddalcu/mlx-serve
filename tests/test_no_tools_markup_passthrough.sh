#!/bin/bash
# A request WITHOUT `tools` gets the model's text as-is, like vLLM/SGLang/
# llama.cpp/Ollama: tool markup the model emits anyway is content, and stream
# and non-stream return the same bytes on every surface.
#
# The system prompt lists a tool in LFM's own format so the model emits
# `<|tool_call_start|>[...]<|tool_call_end|>` with no `tools` field.
#
# Usage: ./tests/test_no_tools_markup_passthrough.sh [model_dir] [port]

set -u

MODEL="${1:-$HOME/.mlx-serve/models/LiquidAI/LFM2.5-2.6B-MLX-4bit}"
PORT="${2:-11262}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG="$HOME/claude-tmp/test_no_tools_markup_passthrough.log"
PASS=0
FAIL=0

check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi
}

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL"
    exit 0
fi

mkdir -p "$(dirname "$LOG")"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 1
"$BINARY" --model "$MODEL" --serve --port "$PORT" --host 127.0.0.1 --ctx-size 8192 > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for _ in $(seq 1 60); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf "$BASE/health" >/dev/null 2>&1 || { echo "FAIL: server did not come up"; exit 1; }

SYS='List of tools: [{"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]'
USER="What's the weather in Paris right now? Use the tool."
MARK='<|tool_call_start|>'

post() { curl -s "$BASE$1" -H 'Content-Type: application/json' -d "$2"; }

sse_text() {
    python3 -c '
import json, sys
kind = sys.argv[1]
out = ""
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    d = json.loads(line[6:])
    if kind == "chat":
        for c in d.get("choices", []):
            out += c["delta"].get("content") or ""
    elif kind == "messages" and d.get("type") == "content_block_delta":
        out += d["delta"].get("text") or ""
    elif kind == "responses" and d.get("type") == "response.output_text.delta":
        out += d["delta"]
sys.stdout.write(out)' "$1"
}

compare() {
    local name="$1" ns="$2" st="$3"
    case "$ns" in *"$MARK"*) check "$name non-stream keeps the markup" 1 ;; *) check "$name non-stream keeps the markup (got: $ns)" 0 ;; esac
    [ "$ns" = "$st" ] && check "$name stream == non-stream" 1 || check "$name stream == non-stream (stream: $st)" 0
}

CHAT=$(jq -nc --arg s "$SYS" --arg u "$USER" '{model:"x",temperature:0,max_tokens:200,messages:[{role:"system",content:$s},{role:"user",content:$u}]}')
MSG=$(jq -nc --arg s "$SYS" --arg u "$USER" '{model:"x",temperature:0,max_tokens:200,system:$s,messages:[{role:"user",content:$u}]}')
RESP=$(jq -nc --arg s "$SYS" --arg u "$USER" '{model:"x",temperature:0,max_output_tokens:200,instructions:$s,input:$u}')

echo "1. /v1/chat/completions"
NS=$(post /v1/chat/completions "$CHAT" | jq -r '.choices[0].message.content')
ST=$(post /v1/chat/completions "$(jq -c '.stream=true' <<<"$CHAT")" | sse_text chat)
compare chat "$NS" "$ST"

echo "2. /v1/messages"
NS=$(post /v1/messages "$MSG" | jq -r '[.content[] | select(.type=="text") | .text] | join("")')
ST=$(post /v1/messages "$(jq -c '.stream=true' <<<"$MSG")" | sse_text messages)
compare messages "$NS" "$ST"

echo "3. /v1/responses"
NS=$(post /v1/responses "$RESP" | jq -r '[.output[] | select(.type=="message") | .content[].text] | join("")')
ST=$(post /v1/responses "$(jq -c '.stream=true' <<<"$RESP")" | sse_text responses)
compare responses "$NS" "$ST"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
