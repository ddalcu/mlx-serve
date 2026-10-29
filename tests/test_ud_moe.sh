#!/usr/bin/env bash
# test_ud_moe.sh — Unsloth UD MoE packs load and generate (bf16 router / shared_expert_gate on a quantized trunk).
#
#   UD_MOE_MODEL=<dir> PORT=8090 ./tests/test_ud_moe.sh   # default unsloth/Qwen3.6-27B-UD-MLX-4bit; SKIPs without

set -euo pipefail

MODEL="${UD_MOE_MODEL:-$HOME/.mlx-serve/models/unsloth/Qwen3.6-27B-UD-MLX-4bit}"
PORT="${PORT:-8090}"
HOST="127.0.0.1"

if [ ! -d "$MODEL" ]; then
    echo "skip: $MODEL not found"
    exit 0
fi

if [ ! -x ./zig-out/bin/mlx-serve ]; then
    echo "FAIL: ./zig-out/bin/mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

LOG=$(mktemp -t mlx-ud-moe.XXXXXX.log)
echo "[ud-moe] starting server: model=$MODEL port=$PORT log=$LOG"
./zig-out/bin/mlx-serve --model "$MODEL" --serve --host "$HOST" --port "$PORT" --no-vision --log-level info >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill -9 $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

# Wait up to 120s for /health
for _ in $(seq 1 120); do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "FAIL: server crashed during load"
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

echo "[ud-moe] server healthy, sending chat completion"

resp=$(curl -sf "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local",
        "messages": [{"role": "user", "content": "Reply with one short greeting."}],
        "max_tokens": 30,
        "temperature": 0.0,
        "stream": false
    }')

content=$(printf '%s' "$resp" | python3 -c "
import json, sys
d = json.load(sys.stdin)
c = d['choices'][0]['message']['content']
print(c)
")

if [ -z "${content// /}" ]; then
    echo "FAIL: empty completion"
    echo "  full response: $resp"
    exit 1
fi

echo "[ud-moe] PASS — generated: $content"
exit 0
