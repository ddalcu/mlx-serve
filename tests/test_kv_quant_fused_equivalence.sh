#!/bin/bash
# Fused quant-attention equivalence test (Plan ricky Phase 2).
#
# Verifies that running the same greedy chat-completion request against the
# server with `--kv-attn-mode dense` (default) and `--kv-attn-mode fused`
# produces the same first-N tokens at `--kv-quant 4`. Both paths read the
# same quantized K/V triples; the only difference is whether SDPA dequantizes
# into a dense tensor first (dense) or consumes the triples directly via
# mlx_quantized_matmul (fused).
#
# The fused path uses Apple's `mlx_quantized_matmul` for the Q@K^T and
# attn@V steps and `mlx_softmax` between. Reduction order differs from the
# fused flash-attention kernel, so the first-N bar exists to catch real
# bugs (wrong transpose, wrong group_size, NaN propagation, GQA broadcast
# mistakes), not to enforce bit-identity.
#
# Threshold:
#   $KV_FUSED_FIRST_N tokens must match (default 25). The fused path
#   reorders reductions vs. the flash-attention kernel inside
#   mlx_fast_scaled_dot_product_attention; observed drift on
#   gemma-4-e4b-it-4bit + --kv-quant 4 is ~26 tokens out (vs. 30 for
#   --kv-quant 4 itself vs. --kv-quant off). 25 is the catch-real-bugs
#   bar (wrong transpose, NaN propagation, GQA broadcast mistakes) —
#   raise per-arch via env var for stricter testing.
#
# Usage:
#   ./tests/test_kv_quant_fused_equivalence.sh [/path/to/model] [port]

set -e

MODEL="${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}"
PORT="${2:-8094}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_kv_quant_fused_equivalence: $MODEL not found."
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model dir."
    exit 1
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

# Long-greedy memorized prompt (same as test_kv_quant_equivalence.sh).
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
    # Args: label, extra-flags, out-var-completion, out-var-tokens
    local label="$1" extra="$2" out_compl="$3" out_tok="$4"
    echo "  starting server ($label)..." >&2
    local logfile
    logfile=$(mktemp)
    "$BINARY" --model "$MODEL" --serve --port "$PORT" --kv-quant 4 --no-pld $extra ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$logfile" 2>&1 &
    local pid=$!
    local up=0 i
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
    printf -v "$out_compl" '%s' "$completion"
    printf -v "$out_tok" '%s' "$tokens"
}

compare_first_n_tokens() {
    local label="$1" tokens_ref="$2" tokens_cand="$3" n="$4"
    local result
    result=$(python3 - <<PY
ref = "$tokens_ref".split(",") if "$tokens_ref" else []
cand = "$tokens_cand".split(",") if "$tokens_cand" else []
n = $n
a = ref[:n]
b = cand[:n]
if len(a) < n or len(b) < n:
    print(f"SHORT len(ref)={len(ref)} len(cand)={len(cand)} need>={n}")
else:
    diverge = -1
    for i,(x,y) in enumerate(zip(a,b)):
        if x != y:
            diverge = i
            break
    if diverge < 0:
        print("OK")
    else:
        print(f"DIFF at index {diverge}: ref={a[diverge]} cand={b[diverge]}")
PY
)
    if [ "$result" = "OK" ]; then
        echo -e "${GREEN}PASS${NC} $label: first $n tokens byte-identical"
        return 0
    else
        echo -e "${RED}FAIL${NC} $label: $result"
        echo "  ref  first $n: $(echo "$tokens_ref" | cut -d',' -f1-$n)"
        echo "  cand first $n: $(echo "$tokens_cand" | cut -d',' -f1-$n)"
        return 1
    fi
}

echo "== Fused quant-attention equivalence =="
echo "  model: $MODEL"
echo "  --kv-quant 4, comparing dense vs fused attention path"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

DENSE_COMPL=""
DENSE_TOK=""
run_and_tokenize "dense" "--kv-attn-mode dense" DENSE_COMPL DENSE_TOK || exit 1
N_DENSE=$(echo "$DENSE_TOK" | tr ',' '\n' | wc -l | tr -d ' ')
echo "  dense: $(echo "$DENSE_COMPL" | wc -c) bytes, $N_DENSE tokens"

sleep 2

FUSED_COMPL=""
FUSED_TOK=""
run_and_tokenize "fused" "--kv-attn-mode fused" FUSED_COMPL FUSED_TOK || exit 1
N_FUSED=$(echo "$FUSED_TOK" | tr ',' '\n' | wc -l | tr -d ' ')
echo "  fused: $(echo "$FUSED_COMPL" | wc -c) bytes, $N_FUSED tokens"

echo

if [ -z "$FUSED_COMPL" ]; then
    echo -e "${RED}FAIL${NC} empty completion under fused — likely NaN or kernel error."
    exit 1
fi

FIRST_N="${KV_FUSED_FIRST_N:-25}"
FAIL=0
compare_first_n_tokens "fused vs dense" "$DENSE_TOK" "$FUSED_TOK" "$FIRST_N" || FAIL=1

exit $FAIL
