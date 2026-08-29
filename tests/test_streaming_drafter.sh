#!/bin/bash
# Streaming drafter byte-equivalence test.
#
# Verifies that running the same temp=0 chat completion request with
# `stream: true` against `--drafter <dir>` produces *identical* concatenated
# text to the same request without `--drafter` (= regular streaming).
#
# Mirrors `tests/test_streaming_pld.sh` for the drafter path — same prompt
# (echo-heavy code rename) so the drafter's verify path is well exercised.
#
# Default pair (Apple Silicon, ~3.3 GB peak RSS):
#   target  = ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit
#   drafter = ~/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16
#
# Override with env vars:
#   DRAFTER_TEST_TARGET=/path/to/gemma-4-target
#   DRAFTER_TEST_DRAFTER=/path/to/gemma-4-{E2B,E4B,...}-it-assistant-bf16
#
# Usage:
#   ./tests/test_streaming_drafter.sh [port]
#
# Exits 0 with SKIP if either checkpoint is missing.

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

TARGET="${DRAFTER_TEST_TARGET:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
DRAFTER="${DRAFTER_TEST_DRAFTER:-$HOME/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16}"

if [ ! -d "$TARGET" ]; then
    echo -e "${YELLOW}SKIP${NC} test_streaming_drafter: target directory not found ($TARGET)."
    exit 0
fi
if [ ! -d "$DRAFTER" ]; then
    echo -e "${YELLOW}SKIP${NC} test_streaming_drafter: drafter directory not found ($DRAFTER)."
    exit 0
fi
if [ ! -f "$TARGET/config.json" ] || [ ! -f "$DRAFTER/config.json" ]; then
    echo -e "${RED}FAIL${NC} target or drafter config.json missing."
    exit 1
fi

DRAFTER_TYPE=$(python3 -c "import json; print(json.load(open('$DRAFTER/config.json')).get('model_type',''))")
if [ "$DRAFTER_TYPE" != "gemma4_assistant" ]; then
    echo -e "${RED}FAIL${NC} drafter at $DRAFTER has model_type='$DRAFTER_TYPE', expected 'gemma4_assistant'"
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build with 'zig build -Doptimize=ReleaseFast'."
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
    local label="$1" mode="$2"
    shift 2
    local extra_args=("$@")
    echo "  starting server ($label)..." >&2
    local logfile
    logfile=$(mktemp)
    "$BINARY" --model "$TARGET" --serve --port "$PORT" "${extra_args[@]}" > "$logfile" 2>&1 &
    local pid=$!
    local up=0
    for i in $(seq 1 60); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
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
    if [ "$mode" = "stream" ]; then stream_flag=True; else stream_flag=False; fi

    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '''$PROMPT'''}],
    'max_tokens': 96,
    'temperature': 0.0,
    'stream': $stream_flag,
}))
")

    local body
    if [ "$mode" = "stream" ]; then
        body=$(echo "$payload" | curl -s -N -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" | sse_concat_content)
    else
        body=$(echo "$payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'], end='')")
    fi

    grep -E "drafter accept=|Drafter ready|drafter=enabled" "$logfile" 2>/dev/null | sed 's/^/    /' >&2 || true
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    sleep 1
    rm -f "$logfile"
    echo "$body"
}

echo "== streaming-drafter byte-equivalence test =="
echo "  target:  $TARGET"
echo "  drafter: $DRAFTER"
echo "  prompt:  <echo-heavy code rename>"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

# Reference: regular streaming, no drafter — what the streamed bytes SHOULD look like.
OUT_REGULAR_STREAM=$(run_request "stream, no --drafter" "stream") || exit 1
echo "  baseline streaming output captured ($(echo -n "$OUT_REGULAR_STREAM" | wc -c) bytes)"
sleep 2

# Streaming with --drafter — must match the regular streaming output exactly.
OUT_DRAFTER_STREAM=$(run_request "stream, --drafter" "stream" --drafter "$DRAFTER") || exit 1
echo "  drafter streaming output captured ($(echo -n "$OUT_DRAFTER_STREAM" | wc -c) bytes)"
sleep 2

# Cross-check: non-streaming with --drafter should also match (this is what
# test_drafter_equivalence.sh covers, but a triple-way diff catches more bugs).
OUT_DRAFTER_NONSTREAM=$(run_request "non-stream, --drafter" "nostream" --drafter "$DRAFTER") || exit 1
echo "  drafter non-streaming output captured ($(echo -n "$OUT_DRAFTER_NONSTREAM" | wc -c) bytes)"

if [ "$OUT_REGULAR_STREAM" = "$OUT_DRAFTER_STREAM" ] && [ "$OUT_REGULAR_STREAM" = "$OUT_DRAFTER_NONSTREAM" ]; then
    echo -e "${GREEN}PASS${NC} streaming + non-streaming drafter output is byte-identical to regular streaming"
    exit 0
else
    echo -e "${RED}FAIL${NC} outputs differ:"
    echo "  regular stream:"
    printf '    %s\n' "$OUT_REGULAR_STREAM"
    echo "  drafter stream:"
    printf '    %s\n' "$OUT_DRAFTER_STREAM"
    echo "  drafter non-stream:"
    printf '    %s\n' "$OUT_DRAFTER_NONSTREAM"
    diff <(echo "$OUT_REGULAR_STREAM") <(echo "$OUT_DRAFTER_STREAM") | sed 's/^/    /' || true
    exit 1
fi
