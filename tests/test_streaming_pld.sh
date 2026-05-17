#!/bin/bash
# Streaming PLD byte-equivalence test.
#
# Verifies that running the same temp=0 chat completion request with
# `stream: true` against `--pld` produces *identical* concatenated text to
# the same request without `--pld`. PLD's streaming path uses the new
# StreamingTokenStream adapter; this is the regression check that the
# adapter doesn't drop / duplicate / reorder tokens vs the regular path.
#
# The chosen prompt is echo-heavy on purpose: the model is asked to repeat
# a code snippet with a small rename. That gives PLD's n-gram lookup plenty
# of long matches → high acceptance and a real exercise of the multi-token
# verify path.
#
# Requires:
#   - A built mlx-serve binary (run `zig build -Doptimize=ReleaseFast` first)
#   - Either:
#       PLD_TEST_MODEL set to a model directory, OR
#       a default MLX checkpoint at ~/.mlx-serve/models/gemma-4-e4b-it-8bit
#
# Usage:
#   PLD_TEST_MODEL=/path/to/model ./tests/test_streaming_pld.sh [port]
#
# Exits 0 with a SKIP message if no model is available.

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${PLD_TEST_MODEL:-$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_streaming_pld: model directory not found."
    echo
    echo "  Set PLD_TEST_MODEL to a model directory, or place an MLX checkpoint"
    echo "  at ~/.mlx-serve/models/gemma-4-e4b-it-8bit (the default this test"
    echo "  looks for). PLD works on any model so the choice is arbitrary."
    exit 0
fi

if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

read -r -d '' PROMPT <<'EOF' || true
Repeat the following Python code exactly, but rename the function from `add` to `sum_two`. Output only the code, no commentary.

def add(a, b):
    result = a + b
    return result

print(add(2, 3))
print(add(10, 20))
EOF

# Extract concatenated `delta.content` from an SSE chat-completions stream.
# Uses python3 to walk the data: lines (skipping `[DONE]` and pings).
sse_concat_content() {
    python3 -c '
import sys, json
out = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "):
        continue
    payload = line[6:].strip()
    if payload == "[DONE]" or not payload:
        continue
    try:
        ev = json.loads(payload)
    except Exception:
        continue
    for ch in ev.get("choices", []) or []:
        delta = ch.get("delta", {}) or {}
        text = delta.get("content")
        if isinstance(text, str):
            out.append(text)
sys.stdout.write("".join(out))
'
}

run_request() {
    local label="$1" pld_flag="$2" mode="$3"
    echo "  starting server ($label)..." >&2
    local logfile
    logfile=$(mktemp)
    "$BINARY" --model "$MODEL" --serve --port "$PORT" $pld_flag ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$logfile" 2>&1 &
    local pid=$!
    local up=0
    for i in $(seq 1 60); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then
            up=1
            break
        fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        echo -e "  ${RED}FAIL${NC} server did not become healthy in 60s" >&2
        tail -20 "$logfile" >&2
        kill $pid 2>/dev/null || true
        rm -f "$logfile"
        return 1
    fi

    local stream_flag
    if [ "$mode" = "stream" ]; then
        stream_flag="true"
    else
        stream_flag="false"
    fi

    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '''$PROMPT'''}],
    'max_tokens': 96,
    'temperature': 0.0,
    'stream': $([[ "$stream_flag" = "true" ]] && echo True || echo False),
}))
")

    local body
    if [ "$mode" = "stream" ]; then
        body=$(echo "$payload" | curl -s -N -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" | sse_concat_content)
    else
        body=$(echo "$payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'], end='')")
    fi

    grep -E "pld accept=" "$logfile" 2>/dev/null | sed 's/^/    /' >&2 || true
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    # Give the kernel a moment to fully release the port + KV cache between
    # back-to-back server runs (previous behavior occasionally raced).
    sleep 1
    rm -f "$logfile"
    echo "$body"
}

echo "== streaming-PLD byte-equivalence test =="
echo "  model: $MODEL"
echo "  prompt: <echo-heavy code rename>"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

# Reference run: regular streaming, no PLD. This is what the streamed bytes
# *should* look like.
OUT_REGULAR_STREAM=$(run_request "stream, no --pld" "--no-pld" "stream") || exit 1
echo "  baseline streaming output captured ($(echo -n "$OUT_REGULAR_STREAM" | wc -c) bytes)"

sleep 2

# Streaming with --pld — must match the regular streaming output exactly.
OUT_PLD_STREAM=$(run_request "stream, --pld" "--pld" "stream") || exit 1
echo "  PLD streaming output captured ($(echo -n "$OUT_PLD_STREAM" | wc -c) bytes)"

sleep 2

# Cross-check: non-streaming with --pld should also match (sanity check —
# this is what `test_pld_equivalence.sh` already covers, but a triple-way
# diff catches more bugs at once).
OUT_PLD_NONSTREAM=$(run_request "non-stream, --pld" "--pld" "nostream") || exit 1
echo "  PLD non-streaming output captured ($(echo -n "$OUT_PLD_NONSTREAM" | wc -c) bytes)"

if [ "$OUT_REGULAR_STREAM" = "$OUT_PLD_STREAM" ] && [ "$OUT_REGULAR_STREAM" = "$OUT_PLD_NONSTREAM" ]; then
    echo -e "${GREEN}PASS${NC} streaming + non-streaming PLD output is byte-identical to regular streaming"
    exit 0
else
    echo -e "${RED}FAIL${NC} outputs differ:"
    echo "  regular stream:"
    printf '    %s\n' "$OUT_REGULAR_STREAM"
    echo "  PLD stream:"
    printf '    %s\n' "$OUT_PLD_STREAM"
    echo "  PLD non-stream:"
    printf '    %s\n' "$OUT_PLD_NONSTREAM"
    diff <(echo "$OUT_REGULAR_STREAM") <(echo "$OUT_PLD_STREAM") | sed 's/^/    /' || true
    exit 1
fi
