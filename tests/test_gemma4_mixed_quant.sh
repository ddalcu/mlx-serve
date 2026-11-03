#!/usr/bin/env bash
# Gemma 4 E4B pack whose tensors override the model-wide group size (gs 32 declared,
# gs 64 stored): warmup, a text turn and an image turn must all run with no MLX error.
#
# Usage: tests/test_gemma4_mixed_quant.sh [model_dir] [port]   (MLX_SERVE_BIN picks the binary)
set -uo pipefail
cd "$(dirname "$0")/.."

source tests/_lib_models.sh
MODEL="${1:-${GEMMA4_MIXED_MODEL:-$(find_model zecanard/gemma-4-E4B-it-ultra-uncensored-heretic-MLX-4bit-mixed_4_6)}}"
PORT="${2:-11391}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
HOUSE="tests/fixtures/house.jpeg"

if [ ! -f "$MODEL/config.json" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi

WORK=$(mktemp -d); LOG="$WORK/server.log"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 1
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info > "$LOG" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT

for i in $(seq 1 240); do curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if ! curl -sf "localhost:$PORT/health" >/dev/null 2>&1; then echo "FAIL: server never came up"; tail -20 "$LOG"; exit 1; fi

FAIL=0
# This checkpoint may think unasked: the answer is judged on reasoning + content.
answer() { python3 -c "import sys,json;m=json.load(sys.stdin)['choices'][0]['message'];print((m.get('reasoning_content') or '')+' '+(m.get('content') or ''))" 2>/dev/null; }

echo "== [1/3] warmup runs the per-layer embedding with no MLX error =="
grep -q "Warmup failed" "$LOG" && { echo "  FAIL: $(grep -m1 '\[mlx\]' "$LOG")"; FAIL=1; } || echo "  OK: warmup"

echo "== [2/3] text turn =="
T=$(curl -s --max-time 300 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"gemma","max_tokens":200,"temperature":0,"messages":[{"role":"user","content":"What is the capital of France? One word."}]}' | answer)
echo "  -> ${T:0:120}"
echo "$T" | grep -qi "paris" || { echo "  FAIL: expected 'Paris'"; FAIL=1; }

echo "== [3/3] image turn (quantized embed_vision projection) =="
B_HOUSE=$(base64 -i "$HOUSE" | tr -d '\n')
T=$(cat <<EOF | curl -s --max-time 300 "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @- | answer
{"model":"gemma","max_tokens":200,"temperature":0,"messages":[{"role":"user","content":[
 {"type":"text","text":"What is the main subject of this image? One word."},
 {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$B_HOUSE"}}]}]}
EOF
)
echo "  -> ${T:0:120}"
echo "$T" | grep -qiE "house|home|building" || { echo "  FAIL: expected 'house'"; FAIL=1; }

if grep -q '\[mlx\]' "$LOG"; then echo "FAIL: MLX errors in the log:"; grep '\[mlx\]' "$LOG" | head -5; FAIL=1; fi
[ "$FAIL" = 0 ] && echo "PASS" || echo "FAILED"
exit "$FAIL"
