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
#       a default MLX checkpoint at ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit
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

MODEL="${BATCHED_TEST_MODEL:-${PLD_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_batched_equivalence: model directory not found."
    echo "  Set BATCHED_TEST_MODEL or place an MLX checkpoint at"
    echo "  ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit."
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
    # Engagement is only observable AFTER a decode has run.
    if [ "$force_flag" = "1" ] && [ "${IS_GDN:-0}" = "1" ] && ! grep -q "gdn batched decode engaged" "$logfile"; then
        echo -e "  ${RED}FAIL${NC} GatedDeltaNet trunk never entered the batched kernel —" >&2
        echo "    this comparison would be serial-vs-serial and pass for free." >&2
        tail -20 "$logfile" >&2
        kill $pid 2>/dev/null || true
        rm -f "$logfile"
        return 1
    fi
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
    if [ "$force_flag" = "1" ] && [ "${IS_GDN:-0}" = "1" ] && ! grep -q "gdn batched decode engaged" "$logfile"; then
        echo -e "  ${RED}FAIL${NC} GatedDeltaNet trunk never entered the batched kernel (long arm)" >&2
        tail -20 "$logfile" >&2
        kill $pid 2>/dev/null || true
        rm -f "$logfile"
        return 1
    fi
    local tokens
    tokens=$(echo "$tok_payload" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/tokenize" | python3 -c "import sys,json; print(','.join(str(t) for t in json.load(sys.stdin)['tokens']))")
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    printf -v "$out_completion_var" '%s' "$completion"
    printf -v "$out_tokens_var" '%s' "$tokens"
}

# A GatedDeltaNet trunk (qwen3_5 family) routes the batched tick through
# `forwardMoeBatchedDecode`, a DIFFERENT kernel from the standard batched one.
# Without this probe the whole test passes vacuously on such a checkpoint: if
# the batching gate ever stops admitting it, both arms silently become the
# same serial path and every byte matches.
IS_GDN=$(python3 - "$MODEL" <<'PYEOF'
import json, sys, pathlib
try:
    c = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
except Exception:
    print("0"); raise SystemExit
t = c.get("text_config") or c
print("1" if t.get("full_attention_interval", 0) else "0")
PYEOF
)

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
else
    echo -e "${RED}FAIL${NC} first-${FIRST_N_TOKENS}-tokens divergence: $DIVERGENCE"
    echo "  single  first ${FIRST_N_TOKENS}: $(echo "$LONG_SINGLE_TOKS" | cut -d',' -f1-${FIRST_N_TOKENS})"
    echo "  batched first ${FIRST_N_TOKENS}: $(echo "$LONG_BATCHED_TOKS" | cut -d',' -f1-${FIRST_N_TOKENS})"
    exit 1
fi

echo
echo "== real N=2 concurrency (batch != 1) =="
# Everything above forces the batched kernel at N=1, which is the ONE width
# where `batch == 1` still holds inside the forward — and several decisions key
# on exactly that. `attnProj` passes `batch == 1 and !is_prefill` as its
# decode_shape, so `--decode-attn-quant` (default ON) engages at forced-N=1 and
# does NOT at real N>1; the fused QK-norm+RoPE gates are `batch == 1` too. A
# guard that only ever runs at N=1 therefore pins a shape that never ships.
# This arm runs two genuinely concurrent streams and holds each to the same
# first-N-tokens bar against the serial answer.
sleep 2
CONC_LOG=$(mktemp)
"$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --max-concurrent 4 > "$CONC_LOG" 2>&1 &
CONC_PID=$!
up=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} concurrent server did not become healthy in 60s"
    tail -20 "$CONC_LOG"; kill $CONC_PID 2>/dev/null || true; rm -f "$CONC_LOG"; exit 1
fi

CONC_A=$(mktemp); CONC_B=$(mktemp)
echo "$LONG_JSON_PAYLOAD" | curl -s -m 180 -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" > "$CONC_A" &
CA=$!
echo "$LONG_JSON_PAYLOAD" | curl -s -m 180 -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" > "$CONC_B" &
CB=$!
wait $CA; wait $CB

# Engagement: output equality alone cannot tell a batched run from two serial
# ones, and two concurrent requests are not guaranteed to overlap. The log line
# is the only proof the batched path ran at N>1.
if ! grep -qE "\[batched\] (gdn batched decode|batched decode) engaged \(slots=[2-9]" "$CONC_LOG"; then
    echo -e "  ${YELLOW}NOT RUN${NC} the two requests never overlapped into a batch of >= 2"
    grep "\[batched\]" "$CONC_LOG" | head -3 | sed 's/^/    /'
    CONC_SKIPPED=1
fi

if [ -z "${CONC_SKIPPED:-}" ]; then
    CONC_FAIL=0
    for f in "$CONC_A" "$CONC_B"; do
        txt=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$f" 2>/dev/null || true)
        if [ -z "$txt" ]; then
            echo -e "${RED}FAIL${NC} concurrent request returned no completion"; cat "$f"; CONC_FAIL=1; break
        fi
        toks=$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1]}))" "$txt" |
            curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/tokenize" |
            python3 -c "import sys,json; print(','.join(str(t) for t in json.load(sys.stdin)['tokens']))")
        verdict=$(python3 - "$LONG_SINGLE_TOKS" "$toks" "$FIRST_N_TOKENS" <<'PYEOF'
import sys
a = sys.argv[1].split(",") if sys.argv[1] else []
b = sys.argv[2].split(",") if sys.argv[2] else []
n = min(int(sys.argv[3]), len(a), len(b))
for i in range(n):
    if a[i] != b[i]:
        print(f"DIFF at index {i}: single={a[i]} concurrent={b[i]}")
        break
else:
    print("OK" if n > 0 else "EMPTY")
PYEOF
)
        if [ "$verdict" != "OK" ]; then
            # A batch of >= 2 is not bit-identical to serial (B=2 matmul /
            # recurrence tiles accumulate in a different order; a hybrid's
            # prefix-cache restore adds its own class). Acquit a divergence
            # whose serial top-2 gap is a near-tie (<= 0.15 nats, the MTP
            # equivalence bar); anything wider is a real bug.
            idx=$(echo "$verdict" | sed -n 's/DIFF at index \([0-9]*\):.*/\1/p')
            gap=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['logprobs']=True; d['top_logprobs']=2; print(json.dumps(d))" "$LONG_JSON_PAYLOAD" |
                curl -s -m 180 -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" |
                python3 -c "
import sys,json
c=json.load(sys.stdin)['choices'][0]['logprobs']['content']; i=int(sys.argv[1])
t=c[i]['top_logprobs']; print(round(t[0]['logprob']-t[1]['logprob'],4))" "$idx" 2>/dev/null || echo 99)
            if python3 -c "import sys; sys.exit(0 if float('$gap') <= 0.15 else 1)"; then
                echo -e "  ${YELLOW}near-tie${NC} $verdict acquitted: serial top-2 gap at index $idx = $gap nats"
            else
                echo -e "${RED}FAIL${NC} N=2 concurrent stream diverged from serial: $verdict (serial top-2 gap $gap nats)"
                CONC_FAIL=1; break
            fi
        fi
    done
    if [ "$CONC_FAIL" != "0" ]; then
        tail -20 "$CONC_LOG"
        kill $CONC_PID 2>/dev/null || true; rm -f "$CONC_LOG" "$CONC_A" "$CONC_B"; exit 1
    fi
    echo -e "${GREEN}PASS${NC} both concurrent streams match serial for ${FIRST_N_TOKENS} tokens (batch >= 2; near-ties acquitted)"
fi
kill $CONC_PID 2>/dev/null || true
wait $CONC_PID 2>/dev/null || true
rm -f "$CONC_LOG" "$CONC_A" "$CONC_B"

echo
echo "== batched-kernel x kv-quant crash guard =="
# Regression: forwardBatchedDecode read cache.entries[].key_view/value_view
# RAW. Under --kv-quant those hold the packed quantized words (hd 256 at
# 8-bit -> last dim 64), so the first concurrent decode fed them straight
# into SDPA and the MLX shape error killed the whole server:
#   [scaled_dot_product_attention] ... query shape (1,8,1,256) for keys
#   shape (1,1,22,64)
# Attention must read KVCache.denseView (the kv-quant contract). Forcing the
# batched kernel at N=1 reproduces it deterministically, no race needed.

sleep 2
KVQ_LOG=$(mktemp)
MLX_SERVE_FORCE_BATCHED=1 "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --kv-quant 8 > "$KVQ_LOG" 2>&1 &
KVQ_PID=$!
up=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then
        up=1
        break
    fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} kv-quant server did not become healthy in 60s"
    tail -20 "$KVQ_LOG"
    kill $KVQ_PID 2>/dev/null || true
    rm -f "$KVQ_LOG"
    exit 1
fi

KVQ_BODY=$(echo "$JSON_PAYLOAD" | curl -s -m 60 -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" || true)
KVQ_CONTENT=$(echo "$KVQ_BODY" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || true)

KVQ_OK=1
if [ -z "$KVQ_CONTENT" ]; then
    echo -e "${RED}FAIL${NC} kv-quant batched request returned no completion"
    echo "  body: $KVQ_BODY"
    KVQ_OK=0
elif ! curl -s -f "$BASE/health" > /dev/null 2>&1; then
    echo -e "${RED}FAIL${NC} server died after kv-quant batched request"
    KVQ_OK=0
elif grep -q "MLX error" "$KVQ_LOG"; then
    echo -e "${RED}FAIL${NC} MLX error in server log during kv-quant batched decode"
    KVQ_OK=0
fi

if [ "$KVQ_OK" != "1" ]; then
    tail -20 "$KVQ_LOG"
    kill $KVQ_PID 2>/dev/null || true
    rm -f "$KVQ_LOG"
    exit 1
fi

kill $KVQ_PID 2>/dev/null || true
wait $KVQ_PID 2>/dev/null || true
rm -f "$KVQ_LOG"
echo -e "${GREEN}PASS${NC} batched decode survives --kv-quant 8 (server alive, completion returned)"

# ── pad-waste cap arm (opt-in: MLX_SERVE_PADWASTE_ARM=1) ──────────────────
# Everything above runs streams of comparable length, where the pad-waste cap
# never has anything to veto. This arm runs the pair the cap EXISTS for: one
# ~1k-token stream beside one ~64k one. `padAndStackBatchedKV` pads every slot
# to the group's longest, so batching those two makes the 1k slot build a
# 64k-wide KV tensor every tick — a per-tick transient no gate bills, whose
# failure mode is an uncatchable Metal OOM.
#
# It was dead on every trunk with a LINEAR global layer 0 (GDN qwen3_5 /
# qwen4_exp, gated-conv lfm2, mamba2 nemotron_h, KDA bailing_hybrid): the cap
# was fed `cache.step`, which only advances inside `update` on layer 0 and so
# reads 0 forever there. Every slot reported 0, the waste ratio was 1.0 for any
# group, and the cap kept everybody — silently, because the output is fine.
#
# Off by default: it needs a long-context checkpoint and a multi-minute 64k
# prefill. The bar is the SPLIT, not bytes — batching decisions on these archs
# change by design, so byte-equality across a changed group would be the wrong
# bar (the forced-N=1 arm and the N=2 near-tie arm above carry that).
if [ "${MLX_SERVE_PADWASTE_ARM:-0}" = "1" ]; then
    echo
    echo "== pad-waste cap: a 1k stream must NOT batch with a 64k one =="

    PROMPTS_DIR="${PADWASTE_PROMPTS_DIR:-$HOME/claude-tmp/bench-qwen4-ladder/prompts_judge}"
    PW_LOG=$(mktemp)
    PW_LONG=$(mktemp); PW_SHORT=$(mktemp)
    PW_LONG_BODY=$(mktemp); PW_SHORT_BODY=$(mktemp)
    PW_PID=""
    cleanup_padwaste() {
        if [ -n "$PW_PID" ]; then
            kill $PW_PID 2>/dev/null || true
            wait $PW_PID 2>/dev/null || true
        fi
        rm -f "$PW_LOG" "$PW_LONG" "$PW_SHORT" "$PW_LONG_BODY" "$PW_SHORT_BODY"
        return 0
    }

    # Request bodies: the bench corpus rungs when they are there (already
    # OpenAI chat bodies), else synthetic filler of the same token order.
    python3 - "$PROMPTS_DIR" "$PW_LONG" "$PW_SHORT" <<'PW_PYEOF'
import json, pathlib, sys

d = pathlib.Path(sys.argv[1])


def body(rung, words, max_tokens):
    src = d / ("rung_%s.json" % rung)
    if src.is_file():
        b = json.loads(src.read_text())
    else:
        # ~1.3 tokens per word of prose-shaped filler.
        filler = " ".join("section %d paragraph body text" % i for i in range(words))
        b = {"messages": [{"role": "user", "content": filler}]}
    b["model"] = "mlx-serve"
    b["max_tokens"] = max_tokens
    b["temperature"] = 0.0
    b["stream"] = True
    b.pop("enable_thinking", None)
    b.pop("enable_pld", None)
    return b


pathlib.Path(sys.argv[2]).write_text(json.dumps(body("64k", 50000, 600)))
pathlib.Path(sys.argv[3]).write_text(json.dumps(body("1k", 800, 400)))
PW_PYEOF

    "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --max-concurrent 4 > "$PW_LOG" 2>&1 &
    PW_PID=$!
    up=0
    for i in $(seq 1 90); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        echo -e "${RED}FAIL${NC} pad-waste server did not become healthy in 90s"
        tail -20 "$PW_LOG"; cleanup_padwaste; exit 1
    fi

    # The long stream first, and the short one only once the long one is
    # DECODING — otherwise the short request finishes during the 64k prefill
    # and the two never form a group at all. Streaming makes "decoding" an
    # observable: the first SSE bytes land when the first token is emitted.
    curl -s -N -m 900 -X POST -H "Content-Type: application/json" -d @"$PW_LONG" \
        "$BASE/v1/chat/completions" > "$PW_LONG_BODY" &
    PW_A=$!
    decoding=0
    for i in $(seq 1 900); do
        if [ -s "$PW_LONG_BODY" ]; then decoding=1; break; fi
        if ! kill -0 $PW_A 2>/dev/null; then break; fi
        sleep 1
    done
    if [ "$decoding" != "1" ]; then
        echo -e "  ${YELLOW}NOT RUN${NC} the 64k stream never reached decode (prefill refused, or too slow)"
        grep -iE "context|memory|refus|error" "$PW_LOG" | head -5 | sed 's/^/    /'
        cleanup_padwaste
        exit 0
    fi

    curl -s -N -m 300 -X POST -H "Content-Type: application/json" -d @"$PW_SHORT" \
        "$BASE/v1/chat/completions" > "$PW_SHORT_BODY" &
    PW_B=$!
    wait $PW_B 2>/dev/null || true
    wait $PW_A 2>/dev/null || true

    PW_FAIL=0
    # (a) the cap must have fired on the pair, naming the waste it compared.
    if ! grep -qE "\[batched\] pad-waste cap: kept [0-9]+ of [0-9]+ slots \(waste " "$PW_LOG"; then
        echo -e "${RED}FAIL${NC} the 1k+64k pair never hit the pad-waste cap —"
        echo "    on a linear-layer-0 trunk this is the dead-cap defect: every slot"
        echo "    reports cache.step == 0, so the waste ratio is 1.0 and nothing caps."
        grep "\[batched\]" "$PW_LOG" | head -5 | sed 's/^/    /'
        PW_FAIL=1
    else
        echo "  $(grep -m1 'pad-waste cap' "$PW_LOG" | sed 's/^ *//')"
    fi
    # (b) and the pair must NOT have been dispatched as one batch of 2: the
    # long slot falls out of the group and decodes serially.
    if grep -qE "engaged \(slots=2\)" "$PW_LOG"; then
        echo -e "${RED}FAIL${NC} the 1k+64k pair batched anyway (engaged (slots=2)) —"
        echo "    the long slot must fall out of the group and decode serially."
        grep -E "engaged \(slots=" "$PW_LOG" | head -5 | sed 's/^/    /'
        PW_FAIL=1
    fi
    if [ "$PW_FAIL" != "0" ]; then
        tail -20 "$PW_LOG"; cleanup_padwaste; exit 1
    fi
    echo -e "${GREEN}PASS${NC} pad-waste cap split the 1k+64k pair (no batch of 2; long slot serial)"
    cleanup_padwaste
fi

exit 0
