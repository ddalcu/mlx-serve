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
# Fused reads are DECODE-WIDTH (T_q == 1) only since 2026-08-15; the spec-on
# section at the bottom pins that verify widths stay off the composed chain
# (engagement counts + the logged Tq), with greedy equivalence vs dense.
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

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
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
    # Args: label, extra-flags, out-var-completion, out-var-tokens, [payload]
    # (payload defaults to $JSON_PAYLOAD). Spec flags ride in extra-flags —
    # the serial sections pass --no-pld, the spec section leaves PLD on.
    # Side channel: ENGAGED_FUSED / ENGAGED_KERNEL / ENGAGED_SPEC /
    # ENGAGED_TQ are set from the server log before the log is deleted.
    local label="$1" extra="$2" out_compl="$3" out_tok="$4" payload="${5:-$JSON_PAYLOAD}"
    echo "  starting server ($label)..." >&2
    local logfile
    logfile=$(mktemp)
    # MIN_TK=1: the per-layer kv floor is a PERF gate; this test's short
    # prompt must still exercise the fused read paths for correctness.
    MLX_SERVE_KV_ATTN_MIN_TK=1 "$BINARY" --model "$MODEL" --serve --port "$PORT" --kv-quant 4 $extra ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$logfile" 2>&1 &
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
    body=$(echo "$payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    local completion
    completion=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
    local tok_payload
    tok_payload=$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1]}))" "$completion")
    local tokens
    tokens=$(echo "$tok_payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/tokenize" | python3 -c "import sys,json; print(','.join(str(t) for t in json.load(sys.stdin)['tokens']))")
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    ENGAGED_FUSED=$(grep -c "\[kv-attn\] fused engaged" "$logfile" || true)
    ENGAGED_KERNEL=$(grep -c "\[kv-attn\] decode kernel engaged" "$logfile" || true)
    ENGAGED_VERIFY=$(grep -c "\[kv-attn\] verify kernel engaged" "$logfile" || true)
    ENGAGED_SPEC=$(grep -c "\[spec-stats\] mode=pld" "$logfile" || true)
    ENGAGED_TQ=$(grep "\[kv-attn\] fused engaged" "$logfile" | head -1 | sed -n 's/.*Tq=\([0-9]*\).*/\1/p')
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
run_and_tokenize "dense" "--kv-attn-mode dense --no-pld" DENSE_COMPL DENSE_TOK || exit 1
N_DENSE=$(echo "$DENSE_TOK" | tr ',' '\n' | wc -l | tr -d ' ')
echo "  dense: $(echo "$DENSE_COMPL" | wc -c) bytes, $N_DENSE tokens"
if [ "${ENGAGED_FUSED:-0}" != "0" ] || [ "${ENGAGED_KERNEL:-0}" != "0" ]; then
    echo -e "${RED}FAIL${NC} dense arm shows [kv-attn] engagement — the dense flag never landed (expectNoSpec class)."
    exit 1
fi

sleep 2

FUSED_COMPL=""
FUSED_TOK=""
run_and_tokenize "fused" "--kv-attn-mode fused --no-pld" FUSED_COMPL FUSED_TOK || exit 1
N_FUSED=$(echo "$FUSED_TOK" | tr ',' '\n' | wc -l | tr -d ' ')
echo "  fused: $(echo "$FUSED_COMPL" | wc -c) bytes, $N_FUSED tokens"
if [ "${ENGAGED_FUSED:-0}" = "0" ]; then
    echo -e "${RED}FAIL${NC} fused arm never logged '[kv-attn] fused engaged' — silent fallback to dense reads."
    exit 1
fi
if [ "${ENGAGED_KERNEL:-0}" = "0" ]; then
    echo -e "${RED}FAIL${NC} fused arm never logged '[kv-attn] decode kernel engaged' — decode width fell through to the composed path."
    exit 1
fi
echo "  engagement: fused=$ENGAGED_FUSED kernel=$ENGAGED_KERNEL"

echo

if [ -z "$FUSED_COMPL" ]; then
    echo -e "${RED}FAIL${NC} empty completion under fused — likely NaN or kernel error."
    exit 1
fi

FIRST_N="${KV_FUSED_FIRST_N:-25}"
FAIL=0
compare_first_n_tokens "fused vs dense" "$DENSE_TOK" "$FUSED_TOK" "$FIRST_N" || FAIL=1

# ── Spec-on arm ────────────────────────────────────────────────────────────
# The decode kernel is DECODE-WIDTH (T_q == 1) only. Verify widths (T_q 2..8
# under PLD/MTP) used to fall to the composed qmm chain — 2 packed matmuls x
# every attention layer per spec round at qmm's M 2..7 dead zone, a measured
# 2.1x decode LOSS at 8k+ (2026-08-15) — and are served by their OWN packed
# kernel now (Phase 2, `[kv-attn] verify kernel engaged: Tq=N`). Asserted
# structurally (engagement counts + the decode line's Tq=1 + a verify-kernel
# engagement), not as a tok/s bar: a one-run spec-decode rate is variance
# per /bench rules — the rates are pinned by the same-boot interleaved perf
# A/B, the math by the qkvVerParityCase fixtures, and the eligibility by the
# kvAttnFusedEligible/kvAttnVerifyEligible unit tests.
echo
echo "== Spec-on arm (PLD + fused reads) =="

# Echo-heavy prompt (same shape as test_pld_equivalence.sh): long n-gram
# matches in the prompt drive deep PLD acceptance, so trunk forwards run at
# verify widths — the widths the old gate routed into the composed chain.
read -r -d '' SPEC_PROMPT <<'EOF' || true
Repeat the following Python code exactly, but rename the function from `add` to `sum_two`. Output only the code, no commentary.

def add(a, b):
    result = a + b
    return result

print(add(2, 3))
print(add(10, 20))
EOF

SPEC_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role': 'user', 'content': '''$SPEC_PROMPT'''}],
    'max_tokens': 96,
    'temperature': 0.0,
    'stream': False,
}))
")

sleep 2
SPEC_DENSE_COMPL=""
SPEC_DENSE_TOK=""
run_and_tokenize "spec dense" "--kv-attn-mode dense --pld" SPEC_DENSE_COMPL SPEC_DENSE_TOK "$SPEC_PAYLOAD" || exit 1
if [ "${ENGAGED_SPEC:-0}" = "0" ]; then
    echo -e "${RED}FAIL${NC} spec dense arm never logged '[spec-stats] mode=pld' — speculation did not engage, the arm proves nothing."
    exit 1
fi

sleep 2
SPEC_FUSED_COMPL=""
SPEC_FUSED_TOK=""
run_and_tokenize "spec fused" "--kv-attn-mode fused --pld" SPEC_FUSED_COMPL SPEC_FUSED_TOK "$SPEC_PAYLOAD" || exit 1
if [ "${ENGAGED_SPEC:-0}" = "0" ]; then
    echo -e "${RED}FAIL${NC} spec fused arm never logged '[spec-stats] mode=pld' — speculation did not engage, the arm proves nothing."
    exit 1
fi
if [ "${ENGAGED_FUSED:-0}" = "0" ]; then
    echo -e "${RED}FAIL${NC} spec fused arm never logged '[kv-attn] fused engaged' — silent fallback to dense reads."
    exit 1
fi
if [ "${ENGAGED_TQ:-}" != "1" ]; then
    echo -e "${RED}FAIL${NC} fused engagement logged Tq=${ENGAGED_TQ:-?} — a non-decode width reached the decode arm (verify widths belong to the verify kernel)."
    exit 1
fi
# No verify-kernel assertion HERE: gemma's shapes sit outside the measured
# adoption set (full-attn dk 512 gated out, sliding masks "array"), so zero
# verify engagements is this arch's CORRECT behavior — asserting one would be
# a checkpoint expectation. The engagement guard runs on a qwen-shaped model
# below.
echo "  engagement: spec=$ENGAGED_SPEC fused=$ENGAGED_FUSED kernel=$ENGAGED_KERNEL verify=${ENGAGED_VERIFY:-0} Tq=$ENGAGED_TQ"

compare_first_n_tokens "spec fused vs spec dense" "$SPEC_DENSE_TOK" "$SPEC_FUSED_TOK" "$FIRST_N" || FAIL=1

# ── Verify-kernel engagement arm (qwen-shaped model) ──────────────────────
# The verify kernel's adoption set is dk <= 256 (qwen3_5-family hd 256) —
# the shapes the Phase-2 A/B measured wins on. A dispatch hole here is
# output-invisible (dense fallback is equivalent), so ENGAGEMENT is the
# assertion, per the spec-test rule.
VERIFY_MODEL="${KV_VERIFY_SPEC_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
if [ ! -d "$VERIFY_MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} verify-kernel arm: $VERIFY_MODEL not found (KV_VERIFY_SPEC_MODEL overrides)."
    exit $FAIL
fi
echo
echo "== Verify-kernel arm ($(basename "$VERIFY_MODEL")) =="
SAVED_MODEL="$MODEL"
MODEL="$VERIFY_MODEL"
sleep 2
VER_DENSE_COMPL=""
VER_DENSE_TOK=""
run_and_tokenize "verify dense" "--kv-attn-mode dense --pld" VER_DENSE_COMPL VER_DENSE_TOK "$SPEC_PAYLOAD" || exit 1
if [ "${ENGAGED_SPEC:-0}" = "0" ]; then
    echo -e "${RED}FAIL${NC} verify-arm dense boot never engaged PLD — the arm proves nothing."
    exit 1
fi
sleep 2
VER_FUSED_COMPL=""
VER_FUSED_TOK=""
run_and_tokenize "verify fused" "--kv-attn-mode fused --pld" VER_FUSED_COMPL VER_FUSED_TOK "$SPEC_PAYLOAD" || exit 1
MODEL="$SAVED_MODEL"
if [ "${ENGAGED_SPEC:-0}" = "0" ]; then
    echo -e "${RED}FAIL${NC} verify-arm fused boot never engaged PLD — the arm proves nothing."
    exit 1
fi
if [ "${ENGAGED_VERIFY:-0}" = "0" ]; then
    echo -e "${RED}FAIL${NC} fused arm never logged '[kv-attn] verify kernel engaged' — verify widths silently fell to dense (dispatch hole; output equality cannot see it)."
    exit 1
fi
echo "  engagement: spec=$ENGAGED_SPEC verify=$ENGAGED_VERIFY"
# Verify widths move near-tie argmaxes (the mtp-equivalence rule): on a
# mismatch replay the dense serial arm with logprobs and acquit only when
# the top-2 gap at the first divergent token is <= 0.15 nats.
if ! compare_first_n_tokens "verify-kernel fused vs dense" "$VER_DENSE_TOK" "$VER_FUSED_TOK" "$FIRST_N"; then
    IDX=$(python3 -c "
a='$VER_DENSE_TOK'.split(','); b='$VER_FUSED_TOK'.split(',')
print(next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y))")
    MODEL="$VERIFY_MODEL"
    LP_PAYLOAD=$(echo "$SPEC_PAYLOAD" | python3 -c "import sys,json; d=json.load(sys.stdin); d.update(max_tokens=$IDX+1, logprobs=True, top_logprobs=2); print(json.dumps(d))")
    echo "  starting server (verify tie probe)..." >&2
    MLX_SERVE_KV_ATTN_MIN_TK=1 "$BINARY" --model "$MODEL" --serve --port "$PORT" --kv-quant 4 --kv-attn-mode dense --no-pld > /dev/null 2>&1 &
    TP_PID=$!
    for i in $(seq 1 60); do curl -s -f "$BASE/health" > /dev/null 2>&1 && break; sleep 1; done
    GAP=$(echo "$LP_PAYLOAD" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" | python3 -c "
import sys,json
try:
    t=json.load(sys.stdin)['choices'][0]['logprobs']['content'][$IDX]['top_logprobs']; print(round(t[0]['logprob']-t[1]['logprob'],3))
except Exception: print('none')")
    kill $TP_PID 2>/dev/null; wait $TP_PID 2>/dev/null
    MODEL="$SAVED_MODEL"
    if python3 -c "import sys; g='$GAP'; sys.exit(0 if g!='none' and float(g) <= 0.15 else 1)"; then
        echo -e "${GREEN}PASS${NC} verify-kernel fused vs dense: argmax flip at a near-tie (index $IDX, top-2 gap=$GAP nats)"
    else
        echo -e "${RED}FAIL${NC} verify-kernel fused vs dense: top-2 gap at index $IDX = $GAP — NOT a near-tie"
        FAIL=1
    fi
fi

exit $FAIL
