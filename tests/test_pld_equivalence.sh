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
#       a default MLX checkpoint at ~/.mlx-serve/models/gemma-4-e4b-it-8bit
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
MODEL="${PLD_TEST_MODEL:-$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_pld_equivalence: model directory not found."
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

# Long-greedy memorized prompt: at INT4 the AR-vs-verify quantized matmul kernels
# in MLX produce slightly different float reduction orders, so near-tie argmax
# tokens can flip beyond ~30–80 tokens. We tolerate that tail by asserting
# byte-equivalence only on the first 30 tokens — that catches real logic
# regressions (wrong argmax from token 0, off-by-one in rollback, etc.) while
# accepting the float-noise cascade. See CLAUDE.md "MTP/PLD/drafter long-greedy
# byte-divergence at INT4".
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

# Number of leading tokens that must match between PLD and AR. Tuned so the
# float-noise tail at INT4 doesn't cause flakes; raise this once an MLX kernel
# fix lands.
FIRST_N_TOKENS=30

run_request() {
    # All status messages go to stderr so the captured stdout is JUST the
    # final completion text from the model. Optional 3rd arg: payload override.
    local label="$1" pld_flag="$2" payload="${3:-$JSON_PAYLOAD}"
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
    local body
    body=$(echo "$payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    grep -E "pld accept=" "$logfile" 2>/dev/null | sed 's/^/    /' >&2 || true
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    echo "$body" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

# Run a full request AND keep the server up so we can hit /tokenize on the same
# server before shutting it down. Writes completion to $1, returns 0/1.
run_and_tokenize() {
    local label="$1" pld_flag="$2" payload="$3" out_completion_var="$4" out_tokens_var="$5"
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
    local body
    body=$(echo "$payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    local completion
    completion=$(echo "$body" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
    local tok_payload
    tok_payload=$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1]}))" "$completion")
    local tokens
    tokens=$(echo "$tok_payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/tokenize" | python3 -c "import sys,json; print(','.join(str(t) for t in json.load(sys.stdin)['tokens']))")
    grep -E "pld accept=" "$logfile" 2>/dev/null | sed 's/^/    /' >&2 || true
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    printf -v "$out_completion_var" '%s' "$completion"
    printf -v "$out_tokens_var" '%s' "$tokens"
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
    echo -e "${GREEN}PASS${NC} short-prompt byte-identical output with vs without --pld"
else
    echo -e "${RED}FAIL${NC} outputs differ:"
    echo "  --no-pld:"
    echo "$OUT_NOPLD" | sed 's/^/    /'
    echo "  --pld:"
    echo "$OUT_PLD" | sed 's/^/    /'
    diff <(echo "$OUT_NOPLD") <(echo "$OUT_PLD") | sed 's/^/    /'
    exit 1
fi

echo
echo "== PLD long-greedy first-${FIRST_N_TOKENS}-tokens equivalence =="
echo "  prompt: <memorized recital, max_tokens=200>"
echo "  rationale: see CLAUDE.md 'MTP/PLD/drafter long-greedy byte-divergence at INT4'"
echo

sleep 2
LONG_COMPLETION_NOPLD=""
LONG_TOKENS_NOPLD=""
run_and_tokenize "without --pld (long)" "--no-pld" "$LONG_JSON_PAYLOAD" LONG_COMPLETION_NOPLD LONG_TOKENS_NOPLD || exit 1
echo "  no-pld long completion ($(echo "$LONG_COMPLETION_NOPLD" | wc -c) bytes, $(echo "$LONG_TOKENS_NOPLD" | tr ',' '\n' | wc -l | tr -d ' ') tokens)"

sleep 2
LONG_COMPLETION_PLD=""
LONG_TOKENS_PLD=""
run_and_tokenize "with --pld (long)" "--pld" "$LONG_JSON_PAYLOAD" LONG_COMPLETION_PLD LONG_TOKENS_PLD || exit 1
echo "  with-pld long completion ($(echo "$LONG_COMPLETION_PLD" | wc -c) bytes, $(echo "$LONG_TOKENS_PLD" | tr ',' '\n' | wc -l | tr -d ' ') tokens)"

# Compare the first FIRST_N_TOKENS tokens. We tolerate divergence past that
# point because of the AR/verify INT4 kernel float-noise tail (see CLAUDE.md).
DIVERGENCE=$(python3 - <<PY
nopld = "$LONG_TOKENS_NOPLD".split(",") if "$LONG_TOKENS_NOPLD" else []
pld   = "$LONG_TOKENS_PLD".split(",") if "$LONG_TOKENS_PLD" else []
n = $FIRST_N_TOKENS
a = nopld[:n]
b = pld[:n]
if len(a) < n or len(b) < n:
    print(f"SHORT len(no-pld)={len(nopld)} len(pld)={len(pld)} need>={n}")
else:
    diverge = -1
    for i,(x,y) in enumerate(zip(a,b)):
        if x != y:
            diverge = i
            break
    if diverge < 0:
        print("OK")
    else:
        print(f"DIFF at index {diverge}: no-pld={a[diverge]} pld={b[diverge]}")
PY
)

if [ "$DIVERGENCE" = "OK" ]; then
    echo -e "${GREEN}PASS${NC} first ${FIRST_N_TOKENS} tokens byte-identical with vs without --pld"
    exit 0
else
    echo -e "${RED}FAIL${NC} first-${FIRST_N_TOKENS}-tokens divergence: $DIVERGENCE"
    echo "  no-pld first ${FIRST_N_TOKENS} tokens: $(echo "$LONG_TOKENS_NOPLD" | cut -d',' -f1-${FIRST_N_TOKENS})"
    echo "  with-pld first ${FIRST_N_TOKENS} tokens: $(echo "$LONG_TOKENS_PLD" | cut -d',' -f1-${FIRST_N_TOKENS})"
    exit 1
fi
