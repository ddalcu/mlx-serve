#!/usr/bin/env bash
# Qwen3-VL image input end-to-end over HTTP (OpenAI + Anthropic surfaces).
# Starts its own server on the Qwen3.5-0.8B 4-bit checkpoint, sends a real image,
# and asserts an HTTP 200 with a relevant, non-empty completion. Also confirms the
# M-RoPE engagement log fires on the chat path. Skips cleanly when the model is
# absent so CI stays green.
#
# Usage: tests/test_qwen_vision.sh [model_dir] [port]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-${QWEN_VISION_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}}"
PORT="${2:-11385}"
IMAGE="tests/fixtures/house.jpeg"

if [ ! -f "$MODEL/config.json" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi
if [ ! -f "$IMAGE" ]; then echo "SKIP: fixture $IMAGE missing"; exit 0; fi

LOG=$(mktemp)
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 1
./zig-out/bin/mlx-serve --model "$MODEL" --serve --port "$PORT" --log-level info > "$LOG" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 90); do curl -s "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if ! curl -s "localhost:$PORT/health" >/dev/null 2>&1; then echo "FAIL: server never came up"; cat "$LOG"; exit 1; fi

B64=$(base64 -i "$IMAGE")
FAIL=0

echo "== OpenAI /v1/chat/completions =="
OAI=$(cat <<EOF | curl -s "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @-
{"model":"qwen","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":[
 {"type":"text","text":"What is the main subject of this image? One word."},
 {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$B64"}}]}]}
EOF
)
OAI_TEXT=$(echo "$OAI" | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  -> $OAI_TEXT"
echo "$OAI_TEXT" | grep -qiE "house|home|building" || { echo "  FAIL: expected 'house' in OpenAI answer"; FAIL=1; }

echo "== Anthropic /v1/messages =="
ANT=$(cat <<EOF | curl -s "localhost:$PORT/v1/messages" -H 'content-type: application/json' -d @-
{"model":"qwen","max_tokens":64,"messages":[{"role":"user","content":[
 {"type":"text","text":"What is the main subject of this image? One word."},
 {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"$B64"}}]}]}
EOF
)
ANT_TEXT=$(echo "$ANT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(''.join(b.get('text','') for b in d.get('content',[])))" 2>/dev/null)
echo "  -> $ANT_TEXT"
echo "$ANT_TEXT" | grep -qiE "house|home|building" || { echo "  FAIL: expected 'house' in Anthropic answer"; FAIL=1; }

echo "== M-RoPE engagement (chat path) =="
grep -qE "M-RoPE: 1 images" "$LOG" && echo "  OK: M-RoPE engaged" || { echo "  FAIL: M-RoPE did not engage"; FAIL=1; }
grep -qE "Qwen grid" "$LOG" && echo "  OK: Qwen ViT ran" || { echo "  FAIL: Qwen encoder did not run"; FAIL=1; }

if [ "$FAIL" = "0" ]; then echo "PASS: qwen vision e2e"; else echo "FAIL: qwen vision e2e"; cat "$LOG" | tail -20; fi
exit $FAIL
