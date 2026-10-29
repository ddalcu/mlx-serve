#!/bin/bash
# test_ane_prefill.sh — ANE prefill offload (--ane-prefill) across boots of one binary: the ready line
# (`[ane] prefill offload ready: N/M mlp + G gdn layers`), per-seam engagement (`[ane] prefill offload engaged`, and
# `[ane] gdn offload engaged` when gdn layers exist: a built program never dispatched is the dispatch-hole class),
# the strictly opt-in off arm (zero `[ane]` lines), banks (strictly fewer banks than covered layers), the
# orphan-staging reap. Every arm asserts ZERO `eval failed` lines: a failed eval recomputes on the GPU, so only the RATE would move.
# Rates are not compared.
#
#   ./tests/test_ane_prefill.sh [/path/to/model] [port]   # default Qwen3.5-0.8B-MLX-4bit (ANE_PREFILL_MODEL)
#
# SKIPs when the private framework is absent. Hermetic twins: unitCount / bankGroupLen / engineBillBytes /
# gateAllows in sdk/ane/offload.zig, anePropsJson.

set -u
source "$(dirname "$0")/lib/server.sh"
MODEL="${1:-${ANE_PREFILL_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}}"
PORT="${2:-8098}"
BASE="http://127.0.0.1:$PORT"
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_ane_prefill: $MODEL not found (ANE_PREFILL_MODEL overrides)."
    exit 0
fi
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

run_arm() {
    # Args: label, flag ("--ane-prefill" or ""), out-var for the reply body.
    local label="$1" flag="$2" out_body="$3"
    echo "  starting server ($label)..." >&2
    LOGFILE=$(mktemp)
    # shellcheck disable=SC2086
    "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" $flag \
        --prefix-cache-entries 0 ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
    SRV=$!
    local up=0 i
    srv_wait_health "$BASE" 120 1 "$SRV" && up=1
    if [ "$up" != "1" ]; then
        echo -e "  ${RED}FAIL${NC} server did not become healthy" >&2
        tail -20 "$LOGFILE" >&2
        kill $SRV 2>/dev/null || true
        return 1
    fi
    # /health answers as soon as the socket binds — the model (and the ANE
    # build, minutes cold) is still loading behind it. Wait for the load's
    # own ready line, timeout scaled to the checkpoint size.
    local model_mb ready_secs ready=0
    model_mb=$(du -sm "$MODEL" 2>/dev/null | awk '{print $1}')
    ready_secs=$(( 600 + ${model_mb:-0} / 100 ))
    for i in $(seq 1 $((ready_secs / 3)) ); do
        if grep -q "Model ready (loaded on inference thread)" "$LOGFILE"; then ready=1; break; fi
        if ! kill -0 $SRV 2>/dev/null; then break; fi
        sleep 3
    done
    if [ "$ready" != "1" ]; then
        echo -e "  ${RED}FAIL${NC} model did not finish loading in ${ready_secs}s" >&2
        tail -20 "$LOGFILE" >&2
        kill $SRV 2>/dev/null || true
        return 1
    fi
    # ~8.6k-token prompt: one FULL default-width (8192) prefill chunk, which
    # is the only shape the fixed-size ANE tiles serve.
    local body
    body=$(python3 -c "
import json
words = ' '.join(f'token{i % 977} alpha beta' for i in range(2900))
print(json.dumps({'model': 'mlx-serve', 'max_tokens': 24, 'temperature': 0.0, 'stream': False,
    'messages': [{'role': 'user', 'content': 'Reply with the single word OK after reading: ' + words}]}))
" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    kill $SRV 2>/dev/null || true
    wait $SRV 2>/dev/null || true
    printf -v "$out_body" '%s' "$body"
}

echo "== ANE prefill offload =="
echo "  model: $MODEL"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

# Plant an orphaned staging dir (dead-pid marker): the on-arm's first ANE
# create must reap it — a killed server's staging otherwise leaks 8-20 GB
# per boot and the framework's own writes start failing as bare
# "compile failed: ?" once the disk fills (the 2026-08-18 class).
ORPHAN="${TMPDIR:-/tmp}/CAFE0000DEAD0000$$"
mkdir -p "$ORPHAN" && echo "999999" > "$ORPHAN/msv-ane.pid" && echo "junk" > "$ORPHAN/model.mil"

ON_BODY=""
run_arm "ane on" "--ane-prefill" ON_BODY || exit 1
ON_LOG="$LOGFILE"
if grep -q "\[ane\] unavailable" "$ON_LOG"; then
    echo -e "${YELLOW}SKIP${NC} AppleNeuralEngine framework unavailable on this machine."
    rm -f "$ON_LOG"
    exit 0
fi
if grep -q "\[ane\] --ane-prefill disabled: NAX-class" "$ON_LOG"; then
    echo -e "${YELLOW}SKIP${NC} the server declines ANE prefill on a NAX GPU (MLX_SERVE_ANE_FORCE=1 runs it anyway)."
    rm -f "$ON_LOG"
    exit 0
fi
if ! grep -q "\[ane\] prefill offload ready" "$ON_LOG"; then
    echo -e "${RED}FAIL${NC} no '[ane] prefill offload ready' line — the build silently died:"
    grep "\[ane\]" "$ON_LOG" | head -5
    rm -f "$ON_LOG"
    exit 1
fi
if ! grep -q "\[ane\] prefill offload engaged" "$ON_LOG"; then
    echo -e "${RED}FAIL${NC} offload built but never engaged — dispatch hole (fixed-shape mismatch?):"
    grep "\[ane\]" "$ON_LOG" | head -5
    rm -f "$ON_LOG"
    exit 1
fi
# A failed eval never fails the REQUEST — the seam recomputes that chunk on
# the GPU — so the only evidence is this line. Silence here is the contract.
assert_no_eval_failures() { # $1 label, $2 log
    local n
    n=$(grep -c "eval failed" "$2")
    if [ "$n" != "0" ]; then
        echo -e "${RED}FAIL${NC} $1: $n failed ANE evals — every one silently fell back to a GPU recompute:"
        grep -m 2 "eval failed" "$2"
        rm -f "$2"
        exit 1
    fi
}
assert_no_eval_failures "on arm" "$ON_LOG"
CONTENT=$(echo "$ON_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
RATE=$(echo "$ON_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('timings',{}).get('prompt_per_second',0))" 2>/dev/null)
if [ -z "$CONTENT" ]; then
    echo -e "${RED}FAIL${NC} empty completion under the offload."
    rm -f "$ON_LOG"
    exit 1
fi
if ! python3 -c "import sys; sys.exit(0 if float('$RATE') > 1 else 1)"; then
    echo -e "${RED}FAIL${NC} prefill rate insane ($RATE tok/s)."
    rm -f "$ON_LOG"
    exit 1
fi
# Per-seam engagement: nonzero gdn coverage must dispatch; zero must not.
GDN_LAYERS=$(grep "prefill offload ready" "$ON_LOG" | sed -n 's/.*+ \([0-9][0-9]*\) gdn layers.*/\1/p')
if [ -z "$GDN_LAYERS" ]; then
    echo -e "${RED}FAIL${NC} ready line does not report a gdn layer count:"
    grep "prefill offload ready" "$ON_LOG"
    rm -f "$ON_LOG"
    exit 1
fi
if [ "$GDN_LAYERS" -gt 0 ]; then
    if ! grep -q "\[ane\] gdn offload engaged" "$ON_LOG"; then
        echo -e "${RED}FAIL${NC} $GDN_LAYERS gdn layers built but the gdn seam never engaged — dispatch hole:"
        grep "\[ane\]" "$ON_LOG" | head -5
        rm -f "$ON_LOG"
        exit 1
    fi
else
    if grep -q "\[ane\] gdn offload engaged" "$ON_LOG"; then
        echo -e "${RED}FAIL${NC} zero gdn layers built yet the gdn seam engaged."
        rm -f "$ON_LOG"
        exit 1
    fi
fi
# Banks: the ready line names how many programs the layers were packed
# into, and it must be FEWER than the layers themselves — one handle per
# layer is exactly what the ~121-handle runtime limit refuses.
READY_LINE=$(grep "prefill offload ready" "$ON_LOG" | head -1)
MLP_LAYERS=$(echo "$READY_LINE" | sed -n 's|.* \([0-9][0-9]*\)/[0-9][0-9]* mlp .*|\1|p')
BANKS=$(echo "$READY_LINE" | sed -n 's/.* in \([0-9][0-9]*\) banks.*/\1/p')
UNITS=$(echo "$READY_LINE" | sed -n 's/.*units=\([0-9][0-9]*\).*/\1/p')
if [ -z "$BANKS" ] || [ -z "$UNITS" ] || [ -z "$MLP_LAYERS" ]; then
    echo -e "${RED}FAIL${NC} ready line does not report units/banks/mlp counts:"
    echo "$READY_LINE"
    rm -f "$ON_LOG"
    exit 1
fi
TOTAL_PROGRAMS=$(( MLP_LAYERS + GDN_LAYERS ))
if [ "$BANKS" -lt 1 ] || [ "$BANKS" -ge "$TOTAL_PROGRAMS" ]; then
    echo -e "${RED}FAIL${NC} $TOTAL_PROGRAMS programs landed in $BANKS banks — that is per-layer programs, not banks:"
    echo "$READY_LINE"
    rm -f "$ON_LOG"
    exit 1
fi
echo "  on arm: ready + engaged (units=$UNITS, gdn=$GDN_LAYERS, $TOTAL_PROGRAMS programs in $BANKS banks), prefill=$RATE tok/s, reply: $(echo "$CONTENT" | head -c 40)"
if [ -d "$ORPHAN" ]; then
    echo -e "${RED}FAIL${NC} orphaned staging dir with a dead-pid marker survived the boot (reap missing)."
    rm -rf "$ORPHAN" "$ON_LOG"
    exit 1
fi
echo "  orphan staging reaped"
rm -f "$ON_LOG"

sleep 2
OFF_BODY=""
run_arm "ane off" "" OFF_BODY || exit 1
OFF_LOG="$LOGFILE"
if grep -q "\[ane\]" "$OFF_LOG"; then
    echo -e "${RED}FAIL${NC} off arm shows [ane] activity — the offload is not opt-in:"
    grep "\[ane\]" "$OFF_LOG" | head -3
    rm -f "$OFF_LOG"
    exit 1
fi
echo "  off arm: zero [ane] lines"
rm -f "$OFF_LOG"

echo -e "${GREEN}PASS${NC} ANE prefill offload: build + engagement + opt-in"
exit 0
