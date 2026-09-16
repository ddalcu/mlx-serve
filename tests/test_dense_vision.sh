#!/usr/bin/env bash
# Dense-bf16 vision regression — proves the bf16-dense multimodal projector path
# stays covered. The standard Gemma 4 SigLIP encoder is shared with the quantized
# models (gemma-4-e4b-it-4bit etc.), but those ship a *quantized* projector
# (`embed_vision.embedding_projection.{weight,scales,biases}`). A fully-dense
# bf16 checkpoint (gemma-4-E2B-it-qat-bf16) ships only `.weight` — no scales — so
# the projector takes vision.zig's dense `transpose+matmul` fallback, which the
# quantized fixtures never exercise. mlx-c throws on `mlx_array_ndim` of the
# null-ctx scales handle, so this is the live guard against that regression.
#
# Critically: starts the server WITHOUT --no-vision (the whole point — qat-bf16
# used to require --no-vision because vision init crashed).
#
# Gated on the model being present (CI-friendly skip). Override the model dir
# with DENSE_VISION_MODEL (abs path). One fixture per image format probed.

set -euo pipefail

PORT="${PORT:-8092}"
HOST="127.0.0.1"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-360}"
MODELS_ROOT="${MODELS_ROOT:-$HOME/.mlx-serve/models/mlx-community}"
MODEL="${DENSE_VISION_MODEL:-$MODELS_ROOT/gemma-4-E2B-it-qat-bf16}"
FIXTURES="$(dirname "$0")/fixtures"

if [ ! -x ./zig-out/bin/mlx-serve ]; then
    echo "FAIL: ./zig-out/bin/mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "skip: dense-bf16 vision model not found ($MODEL)"
    exit 0
fi
for f in house.jpeg not_hot_dog_app.webp; do
    if [ ! -f "$FIXTURES/$f" ]; then
        echo "skip: missing fixture $FIXTURES/$f"
        exit 0
    fi
done

LOG=$(mktemp -t mlx-dense-vision.XXXXXX.log)
echo "[dense-vision] === $(basename "$MODEL") (vision ENABLED) ==="
echo "[dense-vision] starting server (log=$LOG)"
# NOTE: deliberately no --no-vision.
./zig-out/bin/mlx-serve --model "$MODEL" --serve --host "$HOST" --port "$PORT" --log-level info >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill -9 $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 "$LOAD_TIMEOUT"); do
    curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "FAIL: server crashed during load (dense vision init?)"; tail -40 "$LOG"; exit 1
    fi
    sleep 1
done
if ! curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "FAIL: did not become healthy within ${LOAD_TIMEOUT}s"; tail -40 "$LOG"; exit 1
fi

# The encoder must have actually initialized (not silently fallen back to text).
if ! grep -q "Vision encoder:" "$LOG"; then
    echo "FAIL: vision encoder did not initialize — check log:"; tail -40 "$LOG"; exit 1
fi
echo "[dense-vision] $(grep 'Vision encoder:' "$LOG" | head -1)"

send_image() {
    local filepath="$1"; local prompt="$2"; local max_tokens="${3:-20}"
    python3 -c "
import base64, json, urllib.request
with open('$filepath','rb') as f: img=f.read()
ext='$filepath'.rsplit('.',1)[-1].lower()
mime={'jpeg':'image/jpeg','jpg':'image/jpeg','png':'image/png','webp':'image/webp'}.get(ext,'image/jpeg')
b64=base64.b64encode(img).decode()
msg={'model':'local','max_tokens':$max_tokens,'temperature':0.0,'stream':False,
     'messages':[{'role':'user','content':[
         {'type':'image_url','image_url':{'url':f'data:{mime};base64,{b64}'}},
         {'type':'text','text':'''$prompt'''}]}]}
req=urllib.request.Request('http://$HOST:$PORT/v1/chat/completions', json.dumps(msg).encode(), {'Content-Type':'application/json'})
print(json.loads(urllib.request.urlopen(req, timeout=180).read())['choices'][0]['message']['content'].strip())
" 2>/dev/null
}

fail=0
check() { # name result pattern
    if grep -qiE "$3" <<< "$2"; then
        echo "  ✓ $1 → $(echo "$2" | head -1 | cut -c1-50)"
    else
        echo "  ✗ $1 → $(echo "$2" | head -1 | cut -c1-50) (expected: $3)"; fail=1
    fi
}

# JPEG through the dense projector.
check "house color (jpeg)" "$(send_image "$FIXTURES/house.jpeg" 'What color is this house? One word only.')" "blue"
# WebP through the dense projector (different decode path, same projector).
check "food (webp)" "$(send_image "$FIXTURES/not_hot_dog_app.webp" 'What food item do you see? One word.')" "hot.?dog|sausage|frank"

if [ "$fail" -ne 0 ]; then
    echo "[dense-vision] FAIL — dense bf16 projector produced wrong recognition"; exit 1
fi
echo "[dense-vision] PASS — dense bf16 vision projector works end-to-end"
exit 0
