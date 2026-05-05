#!/bin/bash
# PLD (Prompt Lookup Decoding) byte-equivalence test.
#
# Verifies that running the same temp=0 chat completion request against the
# server with --pld produces *identical* output text to running it without
# --pld. PLD is model-agnostic, so any MLX checkpoint will do — no special
# weights required.
#
# The chosen prompt is echo-heavy on purpose: the model is asked to repeat a
# code snippet with one small change. That maximizes n-gram match opportunities
# so the PLD path actually gets exercised (high acceptance rate). The test still
# passes if PLD finds zero matches — both runs use the same greedy sampler so
# they must agree regardless. We just want a real workout for PLD when a model
# is available.
#
# Requires:
#   - A built mlx-serve binary (run `zig build -Doptimize=ReleaseFast` first)
#   - Either:
#       PLD_TEST_MODEL set to a model directory, OR
#       a default MLX checkpoint at ~/.mlx-serve/models/Qwen3.5-4B-MLX-4bit
#
# Usage:
#   PLD_TEST_MODEL=/path/to/model ./tests/test_pld_equivalence.sh [port]
#
# Exits 0 with a SKIP message if no model is available.

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Resolve model directory: explicit env var first, else a sensible default.
MODEL="${PLD_TEST_MODEL:-$HOME/.mlx-serve/models/Qwen3.5-4B-MLX-4bit}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_pld_equivalence: model directory not found."
    echo
    echo "  Set PLD_TEST_MODEL to a model directory, or place an MLX checkpoint"
    echo "  at ~/.mlx-serve/models/Qwen3.5-4B-MLX-4bit (the default this test"
    echo "  looks for). PLD works on any model so the choice is arbitrary."
    exit 0
fi

if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi

# Find binary
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

# Echo-heavy prompt: model is asked to repeat a code snippet with one rename.
# That gives PLD's n-gram lookup plenty of long matches in the prompt → high
# acceptance and a real exercise of the draft+verify path.
read -r -d '' PROMPT <<'EOF' || true
Repeat the following Python code exactly, but rename the function from `add` to `sum_two`. Output only the code, no commentary.

def add(a, b):
    result = a + b
    return result

print(add(2, 3))
print(add(10, 20))
EOF

JSON_PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '''$PROMPT'''}],
    'max_tokens': 96,
    'temperature': 0.0,
    'stream': False,
}))
")

run_request() {
    # All status messages go to stderr so the captured stdout is JUST the
    # final completion text from the model.
    local label="$1" pld_flag="$2"
    echo "  starting server ($label)..." >&2
    local logfile
    logfile=$(mktemp)
    "$BINARY" --model "$MODEL" --serve --port "$PORT" $pld_flag > "$logfile" 2>&1 &
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
    local body
    body=$(echo "$JSON_PAYLOAD" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    grep -E "pld accept=" "$logfile" 2>/dev/null | sed 's/^/    /' >&2 || true
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    echo "$body" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

echo "== PLD byte-equivalence test =="
echo "  model: $MODEL"
echo "  prompt: <echo-heavy code rename>"
echo

# Pre-emptively kill any stale server on the test port.
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

OUT_NOPLD=$(run_request "without --pld" "--no-pld") || exit 1
echo "  no-pld output captured ($(echo "$OUT_NOPLD" | wc -c) bytes)"

sleep 2
OUT_PLD=$(run_request "with --pld" "--pld") || exit 1
echo "  with-pld output captured ($(echo "$OUT_PLD" | wc -c) bytes)"

if [ "$OUT_NOPLD" = "$OUT_PLD" ]; then
    echo -e "${GREEN}PASS${NC} byte-identical output with vs without --pld"
    exit 0
else
    echo -e "${RED}FAIL${NC} outputs differ:"
    echo "  --no-pld:"
    echo "$OUT_NOPLD" | sed 's/^/    /'
    echo "  --pld:"
    echo "$OUT_PLD" | sed 's/^/    /'
    diff <(echo "$OUT_NOPLD") <(echo "$OUT_PLD") | sed 's/^/    /'
    exit 1
fi
