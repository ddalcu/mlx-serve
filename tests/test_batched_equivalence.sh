#!/bin/bash
# Batched-kernel byte-equivalence test (Phase A7).
#
# Verifies that the scheduler's batched-decode kernel produces *byte-identical*
# output to the single-slot decode path at temp=0, single client. The default
# scheduler routes single-slot ticks through `runSingleDecodeTick` (the legacy
# path) — that's the auto-gate at `active.len == 1`. Setting
# `MLX_SERVE_FORCE_BATCHED=1` flips the gate so even N=1 routes through
# `forwardBatchedDecode`. If the two paths diverge, this catches it.
#
# Why test it: the batched kernel laid out tensors differently (positions
# stacked into a [N, 1, d] mvm vs a [1, 1, d] mvm), so even at N=1 it exercises
# code that the single-slot path doesn't. Any silent shape/cache/RoPE bug
# that only shows up under the batched kernel will produce divergent token
# IDs, which this test asserts against.
#
# Like `test_pld_equivalence.sh`, we tolerate float-noise tail past
# FIRST_N_TOKENS at INT4 — the AR/verify quantized matmul reduction order
# differs slightly across the two paths, so near-tie argmax tokens can flip
# on long greedy generations. The first ~30 tokens are stable.
#
# Requires:
#   - A built mlx-serve binary (run `zig build -Doptimize=ReleaseFast`)
#   - Either:
#       BATCHED_TEST_MODEL set to a model directory, OR
#       a default MLX checkpoint at ~/.mlx-serve/models/gemma-4-e4b-it-8bit
#
# Usage:
#   BATCHED_TEST_MODEL=/path/to/model ./tests/test_batched_equivalence.sh [port]

set -e

PORT=${1:-8092}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${BATCHED_TEST_MODEL:-${PLD_TEST_MODEL:-$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit}}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_batched_equivalence: model directory not found."
    echo "  Set BATCHED_TEST_MODEL or place an MLX checkpoint at"
    echo "  ~/.mlx-serve/models/gemma-4-e4b-it-8bit."
    exit 0
fi

if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable. Build first."
    exit 1
fi

# Short prompt for the strict byte-identical assertion. The model's response
# is short enough to land entirely within the float-noise-stable window.
read -r -d '' PROMPT <<'EOF' || true
What is 2+2? Answer with just the number, no explanation.
EOF

JSON_PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '''$PROMPT'''}],
    'max_tokens': 32,
    'temperature': 0.0,
    'stream': False,
}))
")

# Long-greedy memorized recital for the first-N-tokens assertion. Same
# rationale as test_pld_equivalence.sh — we accept float-noise divergence
# past the first ~30 tokens but require the prefix to match exactly.
LONG_PROMPT='Recite the first paragraph of "A Tale of Two Cities" by Charles Dickens.'
LONG_JSON_PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '''$LONG_PROMPT'''}],
    'max_tokens': 200,
    'temperature': 0.0,
    'stream': False,
}))
")

FIRST_N_TOKENS=30

run_request() {
    # All progress to stderr; stdout is the captured completion text.
    local label="$1" force_flag="$2" payload="${3:-$JSON_PAYLOAD}"
    echo "  starting server ($label)..." >&2
    local logfile
    logfile=$(mktemp)
    if [ "$force_flag" = "1" ]; then
        MLX_SERVE_FORCE_BATCHED=1 "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld > "$logfile" 2>&1 &
    else
        "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld > "$logfile" 2>&1 &
    fi
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
    # Confirm force_batched state from the log.
    if [ "$force_flag" = "1" ]; then
        if ! grep -q "force_batched=on" "$logfile"; then
            echo -e "  ${RED}FAIL${NC} expected force_batched=on log line not found" >&2
            tail -20 "$logfile" >&2
            kill $pid 2>/dev/null || true
            rm -f "$logfile"
            return 1
        fi
    fi
    local body
    body=$(echo "$payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    echo "$body" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

run_and_tokenize() {
    local label="$1" force_flag="$2" payload="$3" out_completion_var="$4" out_tokens_var="$5"
    echo "  starting server ($label)..." >&2
    local logfile
    logfile=$(mktemp)
    if [ "$force_flag" = "1" ]; then
        MLX_SERVE_FORCE_BATCHED=1 "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld > "$logfile" 2>&1 &
    else
        "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld > "$logfile" 2>&1 &
    fi
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
    body=$(echo "$payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    local completion
    completion=$(echo "$body" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
    local tok_payload
    tok_payload=$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1]}))" "$completion")
    local tokens
    tokens=$(echo "$tok_payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/tokenize" | python3 -c "import sys,json; print(','.join(str(t) for t in json.load(sys.stdin)['tokens']))")
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    printf -v "$out_completion_var" '%s' "$completion"
    printf -v "$out_tokens_var" '%s' "$tokens"
}

echo "== batched-kernel byte-equivalence test =="
echo "  model: $MODEL"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

OUT_SINGLE=$(run_request "single-slot path (default)" "0") || exit 1
echo "  single-slot output captured ($(echo "$OUT_SINGLE" | wc -c) bytes)"

sleep 2
OUT_BATCHED=$(run_request "force-batched path" "1") || exit 1
echo "  force-batched output captured ($(echo "$OUT_BATCHED" | wc -c) bytes)"

if [ "$OUT_SINGLE" = "$OUT_BATCHED" ]; then
    echo -e "${GREEN}PASS${NC} short-prompt byte-identical (single vs batched)"
else
    echo -e "${RED}FAIL${NC} outputs differ:"
    echo "  single-slot:"
    echo "$OUT_SINGLE" | sed 's/^/    /'
    echo "  force-batched:"
    echo "$OUT_BATCHED" | sed 's/^/    /'
    diff <(echo "$OUT_SINGLE") <(echo "$OUT_BATCHED") | sed 's/^/    /'
    exit 1
fi

echo
echo "== batched-kernel long-greedy first-${FIRST_N_TOKENS}-tokens equivalence =="
echo "  prompt: <memorized recital, max_tokens=200>"
echo "  rationale: see CLAUDE.md 'MTP/PLD/drafter long-greedy byte-divergence at INT4'"
echo

sleep 2
LONG_SINGLE_TEXT=""
LONG_SINGLE_TOKS=""
run_and_tokenize "single-slot (long)" "0" "$LONG_JSON_PAYLOAD" LONG_SINGLE_TEXT LONG_SINGLE_TOKS || exit 1
echo "  single long completion ($(echo "$LONG_SINGLE_TEXT" | wc -c) bytes, $(echo "$LONG_SINGLE_TOKS" | tr ',' '\n' | wc -l | tr -d ' ') tokens)"

sleep 2
LONG_BATCHED_TEXT=""
LONG_BATCHED_TOKS=""
run_and_tokenize "force-batched (long)" "1" "$LONG_JSON_PAYLOAD" LONG_BATCHED_TEXT LONG_BATCHED_TOKS || exit 1
echo "  batched long completion ($(echo "$LONG_BATCHED_TEXT" | wc -c) bytes, $(echo "$LONG_BATCHED_TOKS" | tr ',' '\n' | wc -l | tr -d ' ') tokens)"

DIVERGENCE=$(python3 - <<PY
single = "$LONG_SINGLE_TOKS".split(",") if "$LONG_SINGLE_TOKS" else []
batched = "$LONG_BATCHED_TOKS".split(",") if "$LONG_BATCHED_TOKS" else []
n = $FIRST_N_TOKENS
a = single[:n]
b = batched[:n]
if len(a) < n or len(b) < n:
    print(f"SHORT len(single)={len(single)} len(batched)={len(batched)} need>={n}")
else:
    diverge = -1
    for i,(x,y) in enumerate(zip(a,b)):
        if x != y:
            diverge = i
            break
    if diverge < 0:
        print("OK")
    else:
        print(f"DIFF at index {diverge}: single={a[diverge]} batched={b[diverge]}")
PY
)

if [ "$DIVERGENCE" = "OK" ]; then
    echo -e "${GREEN}PASS${NC} first ${FIRST_N_TOKENS} tokens byte-identical (single vs batched)"
    exit 0
else
    echo -e "${RED}FAIL${NC} first-${FIRST_N_TOKENS}-tokens divergence: $DIVERGENCE"
    echo "  single  first ${FIRST_N_TOKENS}: $(echo "$LONG_SINGLE_TOKS" | cut -d',' -f1-${FIRST_N_TOKENS})"
    echo "  batched first ${FIRST_N_TOKENS}: $(echo "$LONG_BATCHED_TOKS" | cut -d',' -f1-${FIRST_N_TOKENS})"
    exit 1
fi
