#!/bin/bash
# KV-cache quantization equivalence test.
#
# Verifies that running the same greedy (temp=0) chat completion request
# against the server with `--kv-quant 4` or `--kv-quant 8` produces the same
# first-N tokens as `--kv-quant off`. KV quant is mathematically lossy, so we
# tolerate divergence past a per-bit-width threshold — same approach as the
# PLD/drafter long-greedy tests at INT4 weights (see CLAUDE.md "PLD/drafter
# long-greedy byte-divergence at INT4").
#
# Thresholds (empirical; override via env vars per architecture):
#   - 4-bit KV: $KV_QUANT_FIRST_N_4BIT tokens must match (default 30)
#   - 8-bit KV: $KV_QUANT_FIRST_N_8BIT tokens must match (default 30)
#
# The 30-token bar catches kernel bugs (wrong dequantize, off-by-one in
# slice_update, NaN propagation). Beyond ~30 tokens at INT4 *weights* the
# long-greedy float-reduction noise tail dominates regardless of KV bits,
# so a stricter bar would test float-reduction stability, not the quant
# code path. See CLAUDE.md "PLD/drafter long-greedy byte-divergence at INT4".
#
# Empirical per-arch numbers (loose first-N where we observed divergence;
# raise via env var for stricter testing on a given family):
#   gemma-4-e4b-it-4bit:   30-token threshold passes for both 4 and 8 bit;
#                          8-bit happens to stay identical out past 60.
#   Qwen3.5-4B-MLX-4bit:   4-bit passes 30; 8-bit diverges around 41 from
#                          MoE + GatedDeltaNet float-reduction order.
#   LFM2.5-350M-MLX-8bit:  4-bit diverges ~12 tokens — override with
#                          KV_QUANT_FIRST_N_4BIT=10 for this family.
#
# Requires a built mlx-serve binary (zig build -Doptimize=ReleaseFast).
# Default model is gemma-4-e4b-it-4bit; pass any model dir as $1:
#   ./tests/test_kv_quant_equivalence.sh [/path/to/model] [port]

set -e

MODEL="${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}"
PORT="${2:-8094}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_kv_quant_equivalence: $MODEL not found."
    echo "  Pass a model dir as the first argument, e.g.:"
    echo "    $0 ~/.mlx-serve/models/Qwen3.5-4B-MLX-4bit"
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

# Long-greedy memorized prompt — same one PLD/drafter equivalence tests use.
PROMPT='Recite the first paragraph of "A Tale of Two Cities" by Charles Dickens.'

JSON_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '''$PROMPT'''}],
    'max_tokens': 200,
    'temperature': 0.0,
    'stream': False,
}))
")

run_and_tokenize() {
    # Start server with kv-quant flag, fire the request, tokenize the
    # completion, return tokens via global vars. (Mirrors PLD test helper.)
    local label="$1" kv_flag="$2" out_completion_var="$3" out_tokens_var="$4"
    echo "  starting server ($label)..." >&2
    local logfile
    logfile=$(mktemp)
    "$BINARY" --model "$MODEL" --serve --port "$PORT" $kv_flag ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$logfile" 2>&1 &
    local pid=$!
    local up=0
    local i
    for i in $(seq 1 60); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then
            up=1
            break
        fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        echo -e "  ${RED}FAIL${NC} server did not become healthy in 60s" >&2
        tail -30 "$logfile" >&2
        kill $pid 2>/dev/null || true
        rm -f "$logfile"
        return 1
    fi
    local body
    body=$(echo "$JSON_PAYLOAD" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    local completion
    completion=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
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

compare_first_n_tokens() {
    local label="$1" tokens_ref="$2" tokens_quant="$3" n="$4"
    local result
    result=$(python3 - <<PY
ref = "$tokens_ref".split(",") if "$tokens_ref" else []
quant = "$tokens_quant".split(",") if "$tokens_quant" else []
n = $n
a = ref[:n]
b = quant[:n]
if len(a) < n or len(b) < n:
    print(f"SHORT len(ref)={len(ref)} len(quant)={len(quant)} need>={n}")
else:
    diverge = -1
    for i,(x,y) in enumerate(zip(a,b)):
        if x != y:
            diverge = i
            break
    if diverge < 0:
        print("OK")
    else:
        print(f"DIFF at index {diverge}: ref={a[diverge]} quant={b[diverge]}")
PY
)
    if [ "$result" = "OK" ]; then
        echo -e "${GREEN}PASS${NC} $label: first $n tokens byte-identical"
        return 0
    else
        echo -e "${RED}FAIL${NC} $label: $result"
        echo "  ref first $n tokens:   $(echo "$tokens_ref" | cut -d',' -f1-$n)"
        echo "  quant first $n tokens: $(echo "$tokens_quant" | cut -d',' -f1-$n)"
        return 1
    fi
}

echo "== KV-cache quantization equivalence =="
echo "  model: $MODEL"
echo "  prompt: <memorized recital, max_tokens=200>"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

REF_COMPLETION=""
REF_TOKENS=""
run_and_tokenize "off" "--kv-quant off" REF_COMPLETION REF_TOKENS || exit 1
N_REF=$(echo "$REF_TOKENS" | tr ',' '\n' | wc -l | tr -d ' ')
echo "  off: $(echo "$REF_COMPLETION" | wc -c) bytes, $N_REF tokens"

sleep 2

Q4_COMPLETION=""
Q4_TOKENS=""
run_and_tokenize "4-bit" "--kv-quant 4" Q4_COMPLETION Q4_TOKENS || exit 1
N_Q4=$(echo "$Q4_TOKENS" | tr ',' '\n' | wc -l | tr -d ' ')
echo "  4-bit: $(echo "$Q4_COMPLETION" | wc -c) bytes, $N_Q4 tokens"

sleep 2

Q8_COMPLETION=""
Q8_TOKENS=""
run_and_tokenize "8-bit" "--kv-quant 8" Q8_COMPLETION Q8_TOKENS || exit 1
N_Q8=$(echo "$Q8_TOKENS" | tr ',' '\n' | wc -l | tr -d ' ')
echo "  8-bit: $(echo "$Q8_COMPLETION" | wc -c) bytes, $N_Q8 tokens"

echo

# Sanity: neither output may be empty or pad-only.
if [ -z "$Q4_COMPLETION" ] || [ -z "$Q8_COMPLETION" ]; then
    echo -e "${RED}FAIL${NC} empty completion under quantized KV — likely NaN or kernel error."
    exit 1
fi

FIRST_N_4BIT="${KV_QUANT_FIRST_N_4BIT:-30}"
FIRST_N_8BIT="${KV_QUANT_FIRST_N_8BIT:-30}"

FAIL=0
compare_first_n_tokens "4-bit vs off" "$REF_TOKENS" "$Q4_TOKENS" "$FIRST_N_4BIT" || FAIL=1
compare_first_n_tokens "8-bit vs off" "$REF_TOKENS" "$Q8_TOKENS" "$FIRST_N_8BIT" || FAIL=1

exit $FAIL
