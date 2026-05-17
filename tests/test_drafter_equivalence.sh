#!/bin/bash
# Gemma 4 assistant drafter byte-equivalence test.
#
# Verifies that running the same temp=0 chat completion request against the
# server with --drafter <dir> produces *identical* output text to running it
# without --drafter. The drafter is Gemma 4-specific, so the target must be
# a Gemma 4 checkpoint paired with the matching assistant drafter.
#
# Default pair (Apple Silicon, ~3.3 GB peak RSS):
#   target  = ~/.mlx-serve/models/gemma-4-e4b-it-8bit
#   drafter = ~/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16
#
# Override with env vars:
#   DRAFTER_TEST_TARGET=/path/to/gemma-4-target
#   DRAFTER_TEST_DRAFTER=/path/to/gemma-4-{E2B,E4B,...}-it-assistant-bf16
#
# Usage:
#   ./tests/test_drafter_equivalence.sh [port]
#
# Exits 0 with a SKIP message if either checkpoint is missing.

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

TARGET="${DRAFTER_TEST_TARGET:-$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit}"
DRAFTER="${DRAFTER_TEST_DRAFTER:-$HOME/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16}"

if [ ! -d "$TARGET" ]; then
    echo -e "${YELLOW}SKIP${NC} test_drafter_equivalence: target directory not found ($TARGET)."
    echo
    echo "  Set DRAFTER_TEST_TARGET to a Gemma 4 checkpoint, or download"
    echo "  ~/.mlx-serve/models/gemma-4-e4b-it-8bit."
    exit 0
fi

if [ ! -d "$DRAFTER" ]; then
    echo -e "${YELLOW}SKIP${NC} test_drafter_equivalence: drafter directory not found ($DRAFTER)."
    echo
    echo "  Set DRAFTER_TEST_DRAFTER to a gemma4_assistant drafter checkpoint, or"
    echo "  download ~/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16."
    exit 0
fi

if [ ! -f "$TARGET/config.json" ] || [ ! -f "$DRAFTER/config.json" ]; then
    echo -e "${RED}FAIL${NC} $TARGET/config.json or $DRAFTER/config.json missing — not a valid pair."
    exit 1
fi

# Cheap sanity check: the drafter's config must be model_type=gemma4_assistant.
# Without this the server would reject the load anyway, but we surface the
# error early with a clearer message.
DRAFTER_TYPE=$(python3 -c "import json; print(json.load(open('$DRAFTER/config.json')).get('model_type',''))")
if [ "$DRAFTER_TYPE" != "gemma4_assistant" ]; then
    echo -e "${RED}FAIL${NC} drafter at $DRAFTER has model_type='$DRAFTER_TYPE', expected 'gemma4_assistant'"
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

# Echo-heavy prompt: model is asked to repeat a code snippet with one rename.
# Same prompt as test_pld_equivalence so we can compare draft acceptance
# numbers between the two speculative-decoding paths on the same model+prompt.
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
    local label="$1"
    shift
    local extra_args=("$@")
    echo "  starting server ($label)..." >&2
    local logfile
    logfile=$(mktemp)
    "$BINARY" --model "$TARGET" --serve --port "$PORT" "${extra_args[@]}" ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$logfile" 2>&1 &
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
    # Surface drafter acceptance stats from the server log so the test
    # operator can see whether the verify path actually ran.
    grep -E "drafter accept=|Drafter ready" "$logfile" 2>/dev/null | sed 's/^/    /' >&2 || true
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    echo "$body" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

echo "== Drafter byte-equivalence test =="
echo "  target:  $TARGET"
echo "  drafter: $DRAFTER"
echo "  prompt:  <echo-heavy code rename>"
echo

# Pre-emptively kill any stale server on the test port.
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

OUT_NODRAFTER=$(run_request "without --drafter") || exit 1
echo "  no-drafter output captured ($(echo "$OUT_NODRAFTER" | wc -c) bytes)"

sleep 2
OUT_DRAFTER=$(run_request "with --drafter" --drafter "$DRAFTER") || exit 1
echo "  with-drafter output captured ($(echo "$OUT_DRAFTER" | wc -c) bytes)"

if [ "$OUT_NODRAFTER" = "$OUT_DRAFTER" ]; then
    echo -e "${GREEN}PASS${NC} byte-identical output with vs without --drafter"
    exit 0
else
    echo -e "${RED}FAIL${NC} outputs differ:"
    echo "  --no-drafter:"
    echo "$OUT_NODRAFTER" | sed 's/^/    /'
    echo "  --drafter:"
    echo "$OUT_DRAFTER" | sed 's/^/    /'
    diff <(echo "$OUT_NODRAFTER") <(echo "$OUT_DRAFTER") | sed 's/^/    /'
    exit 1
fi
