#!/bin/bash
# ANE prefill offload guard (`--ane-prefill`, perf-plan-aug-17 P5 + v2 GDN).
#
# Contract pinned here:
#   1. BUILD — with the flag on and the framework present, the load logs
#      `[ane] prefill offload ready: N/M mlp + G gdn layers` (a machine
#      without the private framework logs `[ane] unavailable` → SKIP).
#   2. ENGAGEMENT — a prompt long enough to produce one full-width prefill
#      chunk logs `[ane] prefill offload engaged` in the arm's OWN log, and
#      the request completes with non-empty output + a sane prefill rate.
#      PER-SEAM: when the ready line reports nonzero gdn layers, the gdn
#      seam must log its own `[ane] gdn offload engaged` line (a built-but-
#      never-dispatched program is the dispatch-hole class); zero gdn
#      layers must show NO gdn engagement.
#   3. GDN KILL LEVER — MLX_SERVE_ANE_GDN=0 builds `+ 0 gdn layers`, never
#      logs gdn engagement, and the MLP seam still engages (the lever is
#      MLP-only mode, not a global off).
#   4. MODE LEVER (A1) — the DEFAULT is channel mode (`mode=channel` on the
#      ready AND engagement lines; measured 2026-08-18: beats row at less
#      than half the ANE bytes), and MLX_SERVE_ANE_MODE=row builds + engages
#      the row split, with the request completing either way.
#   5. OFF ARM — without the flag there are ZERO `[ane]` lines (the offload
#      is strictly opt-in).
#
# Rates are NOT compared here (one-run rates are variance; the adoption
# numbers live in perf-plan-aug-17.md P5). Model default is the small
# qwen3_5 dense checkpoint; ANE_PREFILL_MODEL overrides.
#
# Usage: ./tests/test_ane_prefill.sh [/path/to/model] [port]

set -u
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
    for i in $(seq 1 120); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
        if ! kill -0 $SRV 2>/dev/null; then break; fi
        sleep 1
    done
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
if ! grep "prefill offload ready" "$ON_LOG" | grep -q "mode=channel"; then
    echo -e "${RED}FAIL${NC} default arm did not report mode=channel:"
    grep "prefill offload ready" "$ON_LOG"
    rm -f "$ON_LOG"
    exit 1
fi
echo "  on arm: ready + engaged (mode=channel, gdn=$GDN_LAYERS), prefill=$RATE tok/s, reply: $(echo "$CONTENT" | head -c 40)"
if [ -d "$ORPHAN" ]; then
    echo -e "${RED}FAIL${NC} orphaned staging dir with a dead-pid marker survived the boot (reap missing)."
    rm -rf "$ORPHAN" "$ON_LOG"
    exit 1
fi
echo "  orphan staging reaped"
rm -f "$ON_LOG"

# GDN kill lever: MLP-only mode, never a global off.
if [ "$GDN_LAYERS" -gt 0 ]; then
    sleep 2
    NOGDN_BODY=""
    export MLX_SERVE_ANE_GDN=0
    run_arm "ane on, gdn off" "--ane-prefill" NOGDN_BODY || { unset MLX_SERVE_ANE_GDN; exit 1; }
    unset MLX_SERVE_ANE_GDN
    NOGDN_LOG="$LOGFILE"
    if ! grep "prefill offload ready" "$NOGDN_LOG" | grep -q "+ 0 gdn layers"; then
        echo -e "${RED}FAIL${NC} MLX_SERVE_ANE_GDN=0 still built gdn programs:"
        grep "prefill offload ready" "$NOGDN_LOG"
        rm -f "$NOGDN_LOG"
        exit 1
    fi
    if grep -q "\[ane\] gdn offload engaged" "$NOGDN_LOG"; then
        echo -e "${RED}FAIL${NC} MLX_SERVE_ANE_GDN=0 yet the gdn seam engaged."
        rm -f "$NOGDN_LOG"
        exit 1
    fi
    if ! grep -q "\[ane\] prefill offload engaged" "$NOGDN_LOG"; then
        echo -e "${RED}FAIL${NC} MLX_SERVE_ANE_GDN=0 killed the MLP seam too (it is MLP-only mode, not off):"
        grep "\[ane\]" "$NOGDN_LOG" | head -5
        rm -f "$NOGDN_LOG"
        exit 1
    fi
    echo "  gdn-off arm: 0 gdn layers, mlp still engaged"
    rm -f "$NOGDN_LOG"
fi

# Row-mode lever arm (A1): the non-default split builds + engages; both
# seams must name mode=row (a channel program serving under a row label —
# or the reverse — is the silent-lever class).
sleep 2
CHAN_BODY=""
export MLX_SERVE_ANE_MODE=row
run_arm "ane on, row mode" "--ane-prefill" CHAN_BODY || { unset MLX_SERVE_ANE_MODE; exit 1; }
unset MLX_SERVE_ANE_MODE
CHAN_LOG="$LOGFILE"
if ! grep "prefill offload ready" "$CHAN_LOG" | grep -q "mode=row"; then
    echo -e "${RED}FAIL${NC} MLX_SERVE_ANE_MODE=row did not build row-mode programs:"
    grep "\[ane\]" "$CHAN_LOG" | head -5
    rm -f "$CHAN_LOG"
    exit 1
fi
if ! grep "prefill offload engaged" "$CHAN_LOG" | grep -q "mode=row"; then
    echo -e "${RED}FAIL${NC} row mode built but the MLP seam never engaged (or engaged as channel):"
    grep "\[ane\]" "$CHAN_LOG" | head -5
    rm -f "$CHAN_LOG"
    exit 1
fi
CHAN_GDN=$(grep "prefill offload ready" "$CHAN_LOG" | sed -n 's/.*+ \([0-9][0-9]*\) gdn layers.*/\1/p')
if [ "${CHAN_GDN:-0}" -gt 0 ] && ! grep "gdn offload engaged" "$CHAN_LOG" | grep -q "mode=row"; then
    echo -e "${RED}FAIL${NC} row mode built $CHAN_GDN gdn layers but the gdn seam never engaged:"
    grep "\[ane\]" "$CHAN_LOG" | head -5
    rm -f "$CHAN_LOG"
    exit 1
fi
CHAN_CONTENT=$(echo "$CHAN_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
if [ -z "$CHAN_CONTENT" ]; then
    echo -e "${RED}FAIL${NC} empty completion under row mode."
    rm -f "$CHAN_LOG"
    exit 1
fi
echo "  row arm: ready + engaged (mode=row, gdn=${CHAN_GDN:-0}), reply: $(echo "$CHAN_CONTENT" | head -c 40)"
rm -f "$CHAN_LOG"

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
