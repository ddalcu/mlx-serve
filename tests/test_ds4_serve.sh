#!/usr/bin/env bash
# Serve-mode smoke test for the embedded ds4 engine.
#
# Boots `mlx-serve --model <gguf> --serve` on a free port, hits
# /v1/chat/completions both non-streaming and streaming, asserts the
# response contains coherent text, then shuts the server down.
#
# Skips quietly when no ds4 GGUF is available locally. Set DS4_GGUF=<path>
# to point at a checkpoint, or drop one at the canonical Swift-app path.

set -euo pipefail

BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${PORT:-11288}"

DEFAULT_CANDIDATES=(
    "${DS4_GGUF:-}"
    "$HOME/.mlx-serve/models/antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
    "$HOME/projects/agents/ds4/ds4flash.gguf"
    "$HOME/projects/agents/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
)

GGUF=""
for c in "${DEFAULT_CANDIDATES[@]}"; do
    if [[ -n "$c" && -f "$c" ]]; then
        GGUF="$c"
        break
    fi
done

if [[ -z "$GGUF" ]]; then
    echo "[skip] no ds4 GGUF found locally — set DS4_GGUF=<path> or drop one at ~/.mlx-serve/models/antirez/deepseek-v4-gguf/"
    exit 0
fi

if [[ ! -x "$BINARY" ]]; then
    echo "[fail] mlx-serve binary not found at $BINARY — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[ok] booting ds4 server on port $PORT (gguf: $(basename "$GGUF"))"
"$BINARY" --model "$GGUF" --serve --port "$PORT" --max-tokens 32 >/tmp/test_ds4_serve.log 2>&1 &
SERVER_PID=$!

# Wait for /health to flip green (engine open takes a few seconds).
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[ok] server up after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[fail] server exited during startup; log tail:"
        tail -30 /tmp/test_ds4_serve.log
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[fail] /health never came up; log tail:"
    tail -30 /tmp/test_ds4_serve.log
    exit 1
fi

# Non-streaming chat completion.
echo "[ok] POST /v1/chat/completions (non-streaming)"
RESP=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Write a haiku about Apple Silicon GPUs."}],"max_tokens":32,"temperature":0.0}')

CONTENT=$(printf '%s' "$RESP" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])' 2>/dev/null || true)

if [[ -z "$CONTENT" ]]; then
    echo "[fail] empty content in non-streaming response"
    printf '%s\n' "$RESP" | head -200
    exit 1
fi

WORDS=$(printf '%s' "$CONTENT" | wc -w | tr -d ' ')
if (( WORDS < 3 )); then
    echo "[fail] non-streaming generated only $WORDS word(s):"
    printf '%s\n' "$CONTENT"
    exit 1
fi

echo "[ok] non-streaming generated $WORDS words:"
echo "----"
printf '%s\n' "$CONTENT"
echo "----"

# Streaming chat completion.
echo "[ok] POST /v1/chat/completions (streaming)"
STREAM_OUT=$(curl -sf -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Write a one-line greeting."}],"max_tokens":24,"temperature":0.0,"stream":true}')

# Concatenate non-empty content deltas.
STREAM_CONTENT=$(printf '%s\n' "$STREAM_OUT" \
    | awk '/^data: /{ sub(/^data: /, ""); print }' \
    | python3 -c '
import json, sys
out = []
for line in sys.stdin:
    line = line.strip()
    if not line or line == "[DONE]":
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    for ch in obj.get("choices", []):
        d = ch.get("delta") or {}
        c = d.get("content")
        if c:
            out.append(c)
print("".join(out))
' 2>/dev/null || true)

if [[ -z "$STREAM_CONTENT" ]]; then
    echo "[fail] no streamed content"
    printf '%s\n' "$STREAM_OUT" | head -20
    exit 1
fi

S_WORDS=$(printf '%s' "$STREAM_CONTENT" | wc -w | tr -d ' ')
if (( S_WORDS < 2 )); then
    echo "[fail] streaming generated only $S_WORDS word(s):"
    printf '%s\n' "$STREAM_CONTENT"
    exit 1
fi

echo "[ok] streaming generated $S_WORDS words:"
echo "----"
printf '%s\n' "$STREAM_CONTENT"
echo "----"

echo "[pass] tests/test_ds4_serve.sh"
