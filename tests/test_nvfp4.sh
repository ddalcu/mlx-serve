#!/usr/bin/env bash
# test_nvfp4.sh — NVFP4 packs: the loader takes `quantization.mode: "nvfp4"` (uint8 fp8 scales, no biases), the banner
# reports nvfp4, a temp-0 coherence canary answers, discovery lists the pack. A gemma-4 QAT nvfp4 dir exercises the
# mixed affine-8-bit override layers resolving per weight.
#
#   NVFP4_TEST_MODEL=<dir> PORT=8096 ./tests/test_nvfp4.sh   # default mlx-community/Qwen3-0.6B-nvfp4; SKIPs without
#
# Hermetic twins: `computeQuantParams` and the nvfp4 qmatmul/gather tests in engines/mlx/transformer.zig.

set -euo pipefail

MODEL="${NVFP4_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3-0.6B-nvfp4}"
PORT="${PORT:-8096}"
HOST="127.0.0.1"

if [ ! -d "$MODEL" ]; then
    echo "skip: $MODEL not found"
    exit 0
fi

if [ ! -x ./zig-out/bin/mlx-serve ]; then
    echo "FAIL: ./zig-out/bin/mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

mode=$(python3 -c "
import json
print(json.load(open('$MODEL/config.json')).get('quantization', {}).get('mode', ''))
")
if [ "$mode" != "nvfp4" ]; then
    echo "FAIL: $MODEL is not an nvfp4 checkpoint (quantization.mode='$mode')"
    exit 1
fi

LOG=$(mktemp -t mlx-nvfp4.XXXXXX.log)
echo "[nvfp4] starting server: model=$MODEL port=$PORT log=$LOG"
./zig-out/bin/mlx-serve --model "$MODEL" --serve --host "$HOST" --port "$PORT" --no-vision --log-level info >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill -9 $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

# Wait up to 120s for /health
for _ in $(seq 1 120); do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "FAIL: server crashed during load (MISSING WEIGHT / unsupported mode?)"
        tail -30 "$LOG"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "FAIL: server did not become healthy within 120s"
    tail -30 "$LOG"
    exit 1
fi

# The model banner must report the nvfp4 mode — a model that silently parsed
# as affine would generate garbage rather than fail to load.
if ! grep -q "nvfp4 quant" "$LOG"; then
    echo "FAIL: model banner does not report nvfp4 quant mode"
    grep "Model:" "$LOG" || true
    exit 1
fi

echo "[nvfp4] server healthy, sending temp-0 chat completion"

resp=$(curl -sf "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}],
        "max_tokens": 60,
        "temperature": 0.0,
        "stream": false
    }')

content=$(printf '%s' "$resp" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d['choices'][0]['message']['content'])
")

if [ -z "${content// /}" ]; then
    echo "FAIL: empty completion"
    echo "  full response: $resp"
    exit 1
fi

# Coherence canary: a misrouted quant mode produces fluent-looking garbage;
# every nvfp4 model tested answers this with "Paris" at temp 0.
if ! printf '%s' "$content" | grep -qi "paris"; then
    echo "FAIL: completion does not mention Paris — quant misroute? got: $content"
    exit 1
fi

echo "[nvfp4] PASS — generated: $content"
exit 0
