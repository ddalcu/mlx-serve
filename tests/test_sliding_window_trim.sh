#!/bin/bash
# Sliding-window BLOCK trim byte-equivalence, per arch.
#
# A sliding layer only needs the tail its queries can reach: `window` for a
# decode step, `window + q_len - 1` for a block-wide forward. Decode has always
# trimmed; the block half used to fall through and read the ENTIRE cache on
# every sliding layer (see CLAUDE.md, docs/gotchas/engine-mlx.md).
#
# Trimming the K/V view is only correct if every consumer of that view derives
# its query offset RELATIVELY. This test is the proof: greedy (temperature 0)
# output with MLX_SERVE_SLIDING_BLOCK_TRIM=1 must be BYTE-IDENTICAL to the same
# request with =0, at a context well past the window. A diff means that arch's
# mask or bias reads absolute positions and is not view-relative — report it,
# don't ship it.
#
# Each arm asserts an `[sliding] block trim engaged` line in its OWN log (the
# =0 arm asserts its ABSENCE), so a silently-declined trim reads as a failure
# rather than a pass.
#
# The prompt is sized past window + prefill chunk on purpose: below that the
# span covers the whole cache and no trim happens, so the two arms would agree
# for the wrong reason.
#
# Requires a built binary (`zig build -Doptimize=ReleaseFast`) and at least one
# checkpoint. Every arch is env-gated and skipped when absent.
#
# Usage:
#   SLIDING_GEMMA4_MOE_MODEL=/path SLIDING_LAGUNA_MODEL=/path \
#   SLIDING_INKLING_MODEL=/path SLIDING_GPT_OSS_MODEL=/path \
#   ./tests/test_sliding_window_trim.sh [port]

set -u

PORT=${1:-8098}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

SSD="/Volumes/G Drive SSD/models"
GEMMA4_MOE="${SLIDING_GEMMA4_MOE_MODEL:-$SSD/mlx-community/gemma-4-26b-a4b-it-4bit}"
LAGUNA="${SLIDING_LAGUNA_MODEL:-$SSD/poolside/Laguna-XS-2.1-NVFP4-mlx}"
INKLING="${SLIDING_INKLING_MODEL:-$SSD/mlx-community/Inkling-Small-mlx-2bit}"
GPT_OSS="${SLIDING_GPT_OSS_MODEL:-$HOME/.mlx-serve/models/mlx-community/gpt-oss-20b-MXFP4-Q8}"

FAILURES=0
RAN=0

# Deterministic filler followed by a question whose answer lives at the TOP of
# the prompt — so an attention window that reads the wrong slice of the cache
# changes the answer, not just the phrasing.
#
# Length is per-arch. A prefill CHUNK only takes the trim once it lands at a
# non-zero offset (chunk 1 has total_kv == seq_len, which the span always
# covers), so a serial arch needs a prompt past 2x the default 8192 chunk —
# at 8k it prefills in ONE chunk and the two arms agree for the wrong reason.
build_payload() {
    python3 - "$1" "$2" <<'PY'
import json, sys
max_tokens, sections = int(sys.argv[1]), int(sys.argv[2])
head = "REMEMBER THIS: the vault code is 74-19-32 and the keeper's name is Almeida.\n\n"
para = ("Section {i}. The survey team recorded the following observations at station {i}: "
        "the water level rose by {a} centimetres, the sediment load measured {b} grams per litre, "
        "and the ambient temperature held at {c} degrees. No anomalies were logged. "
        "The equipment was recalibrated before the next reading was taken.\n")
body = "".join(para.format(i=i, a=(i * 7) % 40, b=(i * 13) % 90, c=(i * 3) % 30) for i in range(1, sections + 1))
tail = "\n\nUsing only the text above, state the vault code and the keeper's name, then repeat Section 3 verbatim."
print(json.dumps({
    "model": "mlx-serve",
    "messages": [{"role": "user", "content": head + body + tail}],
    "max_tokens": max_tokens,
    "temperature": 0.0,
    "stream": False,
}))
PY
}

# run_arm <model> <trim 0|1> <extra server args> <max_tokens> <sections> <outfile>
# Writes the completion to $6 and the server log to $6.log. Echoes prompt_tokens.
run_arm() {
    local model="$1" trim="$2" extra="$3" max_tokens="$4" sections="$5" out="$6"
    local logfile="$out.log"
    MLX_SERVE_SLIDING_BLOCK_TRIM="$trim" "$BINARY" --model "$model" --serve --port "$PORT" \
        --prefix-cache-entries 0 $extra > "$logfile" 2>&1 &
    local pid=$!
    local up=0
    for _ in $(seq 1 300); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
        if ! kill -0 $pid 2>/dev/null; then break; fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        echo -e "    ${RED}FAIL${NC} server did not become healthy" >&2
        tail -20 "$logfile" >&2
        kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true
        return 1
    fi
    local body
    body=$(build_payload "$max_tokens" "$sections" | curl -s --max-time 1800 -X POST \
        -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true

    echo "$body" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'choices' not in d:
    sys.stderr.write('server error: %s\n' % json.dumps(d)[:400]); sys.exit(1)
sys.stderr.write('%d\n' % d.get('usage', {}).get('prompt_tokens', 0))
print(d['choices'][0]['message']['content'])
" > "$out" 2> "$out.ptok" || { cat "$out.ptok" >&2; return 1; }
    cat "$out.ptok"
}

check_arch() {
    local label="$1" model="$2" extra="$3" max_tokens="$4" sections="$5"
    if [ ! -d "$model" ]; then
        echo -e "${YELLOW}SKIP${NC} $label: $model not found"
        return 0
    fi
    RAN=$((RAN + 1))
    echo "== $label =="
    echo "  model: $model"
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
    sleep 1

    local tmp; tmp=$(mktemp -d)
    local ptok_off ptok_on
    echo "  arm A: MLX_SERVE_SLIDING_BLOCK_TRIM=0 (decode-width only)"
    ptok_off=$(run_arm "$model" 0 "$extra" "$max_tokens" "$sections" "$tmp/off") || { FAILURES=$((FAILURES+1)); rm -rf "$tmp"; return 0; }
    sleep 3
    echo "  arm B: MLX_SERVE_SLIDING_BLOCK_TRIM=1 (block-wide trim)"
    ptok_on=$(run_arm "$model" 1 "$extra" "$max_tokens" "$sections" "$tmp/on") || { FAILURES=$((FAILURES+1)); rm -rf "$tmp"; return 0; }

    echo "  prompt_tokens: off=$ptok_off on=$ptok_on"

    # Engagement: the ON arm must log the block trim, the OFF arm must not.
    if ! grep -q "\[sliding\] block trim engaged" "$tmp/on.log"; then
        echo -e "  ${RED}FAIL${NC} $label: trim-on arm never engaged the block trim"
        echo "    (prompt too short for this window/chunk, or the arch declined it)"
        FAILURES=$((FAILURES + 1)); rm -rf "$tmp"; return 0
    fi
    grep -m1 "\[sliding\] block trim engaged" "$tmp/on.log" | sed 's/^/    /'
    if grep -q "\[sliding\] block trim engaged" "$tmp/off.log"; then
        echo -e "  ${RED}FAIL${NC} $label: kill switch did not disable the block trim"
        FAILURES=$((FAILURES + 1)); rm -rf "$tmp"; return 0
    fi

    if cmp -s "$tmp/off" "$tmp/on"; then
        echo -e "  ${GREEN}PASS${NC} $label: byte-identical with and without the block trim"
    else
        echo -e "  ${RED}FAIL${NC} $label: output DIFFERS — this arch's mask/bias is not view-relative"
        echo "  --- trim off ---"; head -c 600 "$tmp/off"; echo
        echo "  --- trim on  ---"; head -c 600 "$tmp/on"; echo
        cp "$tmp/off" "/tmp/sliding_${label}_off.txt" 2>/dev/null || true
        cp "$tmp/on" "/tmp/sliding_${label}_on.txt" 2>/dev/null || true
        FAILURES=$((FAILURES + 1))
    fi
    rm -rf "$tmp"
    echo
}

echo "=== sliding-window block trim equivalence ==="
echo

# gemma4 MoE: hd 256, window 1024, 5:1 sliding:full. --pld gives the multi-token
# VERIFY blocks (4-8 wide) this trim is for — below the fused kernel's 16-row
# floor, so the band is NOT in-kernel and the trim applies.
check_arch "gemma4-moe" "$GEMMA4_MOE" "--pld" 128 121

# laguna: hd 128, window 512, serial arch — its only multi-token forward is the
# prefill CHUNK, which is exactly what the removed width cap used to exclude.
check_arch "laguna" "$LAGUNA" "--no-pld" 96 320

# inkling: hd 128, window 512, 35/42 local layers, RelativeLogits bias instead
# of RoPE — the bias is the thing that has to be view-relative here.
check_arch "inkling" "$INKLING" "--no-pld" 96 320

# gpt-oss: hd 64, window 128 — by far the smallest here, so the prompt is a
# fraction of the others': anything past ~130 tokens already makes every PLD
# verify block a trimmed one, and a 20B at this machine's 8k prefill budget is
# what the size is actually bounded by. This arm is the regression test for the
# shape mismatch: the arm sized its K/V view with the raw `cfg.sliding_window`
# while building masks at `window + q_len - 1`, which is invisible at q_len 1
# and an uncatchable MLX error (`[broadcast_shapes] (1,1,6,133) vs
# (1,64,6,128)`) on the first prefill chunk or spec-verify block past the
# window. Before the fix the request kills the server outright, so this reads
# as a failure, not a diff.
check_arch "gpt-oss" "$GPT_OSS" "--pld" 128 24

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true

if [ "$RAN" = "0" ]; then
    echo -e "${YELLOW}SKIP${NC} no checkpoints available"
    exit 0
fi
if [ "$FAILURES" != "0" ]; then
    echo -e "${RED}$FAILURES arch(es) FAILED${NC}"
    exit 1
fi
echo -e "${GREEN}ALL PASS${NC} ($RAN arch(es))"
