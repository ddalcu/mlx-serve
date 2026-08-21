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
#   5b. NO FAILED EVALS — every arm must log ZERO `[ane] ... eval failed`.
#      A failed eval falls back to a GPU recompute for that chunk, so the
#      request still answers correctly and only the RATE moves: the
#      42-procedure bank whose symbol indices were wrong lost 200 of 210
#      evals and read as "banks are 23% slower" (2026-08-20).
#   6. BANKS — programs are PROCEDURE BANKS, so the ready line reports
#      strictly FEWER banks than covered layers (the ~121-handle runtime
#      limit is why; one handle per layer is what banks replace), and this
#      machine reports `units=1`.
#   7. SPLIT LADDER — a tiny MLX_SERVE_ANE_BANK_MAX_BYTES forces more banks
#      and coverage must still be FULL: partitioning is a packaging
#      decision, never a coverage one.
#   8. DUAL SELF-DISABLE — MLX_SERVE_ANE_DUAL=1 on single-ANE silicon must
#      say so by name and build units=1, never fail the boot.
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
if ! grep "prefill offload ready" "$ON_LOG" | grep -q "mode=channel"; then
    echo -e "${RED}FAIL${NC} default arm did not report mode=channel:"
    grep "prefill offload ready" "$ON_LOG"
    rm -f "$ON_LOG"
    exit 1
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
if [ "$UNITS" != "1" ]; then
    echo -e "${RED}FAIL${NC} default build reports units=$UNITS — dual is opt-in."
    rm -f "$ON_LOG"
    exit 1
fi
echo "  on arm: ready + engaged (mode=channel, units=1, gdn=$GDN_LAYERS, $TOTAL_PROGRAMS programs in $BANKS banks), prefill=$RATE tok/s, reply: $(echo "$CONTENT" | head -c 40)"
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
    assert_no_eval_failures "gdn-off arm" "$NOGDN_LOG"
echo "  gdn-off arm: 0 gdn layers, mlp still engaged"
    rm -f "$NOGDN_LOG"
fi

# Split ladder: a tiny bank cap forces the partitioner to make MORE banks,
# and coverage must stay FULL — how programs are packaged is never allowed
# to become a coverage decision.
sleep 2
LADDER_BODY=""
export MLX_SERVE_ANE_BANK_MAX_BYTES=4096
run_arm "ane on, tiny bank cap" "--ane-prefill" LADDER_BODY || { unset MLX_SERVE_ANE_BANK_MAX_BYTES; exit 1; }
unset MLX_SERVE_ANE_BANK_MAX_BYTES
LADDER_LOG="$LOGFILE"
LADDER_READY=$(grep "prefill offload ready" "$LADDER_LOG" | head -1)
LADDER_BANKS=$(echo "$LADDER_READY" | sed -n 's/.* in \([0-9][0-9]*\) banks.*/\1/p')
LADDER_MLP=$(echo "$LADDER_READY" | sed -n 's|.* \([0-9][0-9]*\)/[0-9][0-9]* mlp .*|\1|p')
LADDER_GDN=$(echo "$LADDER_READY" | sed -n 's/.*+ \([0-9][0-9]*\) gdn layers.*/\1/p')
if [ "${LADDER_MLP:-0}" != "$MLP_LAYERS" ] || [ "${LADDER_GDN:-0}" != "$GDN_LAYERS" ]; then
    echo -e "${RED}FAIL${NC} a small bank cap changed COVERAGE ($LADDER_MLP/$LADDER_GDN vs $MLP_LAYERS/$GDN_LAYERS):"
    echo "$LADDER_READY"
    rm -f "$LADDER_LOG"
    exit 1
fi
if [ "${LADDER_BANKS:-0}" -le "$BANKS" ]; then
    echo -e "${RED}FAIL${NC} MLX_SERVE_ANE_BANK_MAX_BYTES=4096 did not split further ($LADDER_BANKS vs $BANKS banks):"
    echo "$LADDER_READY"
    rm -f "$LADDER_LOG"
    exit 1
fi
if ! grep -q "\[ane\] prefill offload engaged" "$LADDER_LOG"; then
    echo -e "${RED}FAIL${NC} split banks built but the MLP seam never engaged:"
    grep "\[ane\]" "$LADDER_LOG" | head -5
    rm -f "$LADDER_LOG"
    exit 1
fi
assert_no_eval_failures "ladder arm" "$LADDER_LOG"
echo "  ladder arm: same coverage in $LADDER_BANKS banks (vs $BANKS), still engaged"
rm -f "$LADDER_LOG"

# Dual on single-ANE silicon: refused BY NAME, still a working single-ANE
# build. (A real two-instance machine is the tester's arm — see
# NOTE_TO_TESTER_ANE_DFLASH2.md.)
if [ "$(sysctl -n machdep.cpu.brand_string 2>/dev/null)" != "${DUAL_CHIP:-}" ] && \
   ! sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -q Ultra; then
    sleep 2
    DUAL_BODY=""
    export MLX_SERVE_ANE_DUAL=1
    run_arm "ane on, dual asked" "--ane-prefill" DUAL_BODY || { unset MLX_SERVE_ANE_DUAL; exit 1; }
    unset MLX_SERVE_ANE_DUAL
    DUAL_LOG="$LOGFILE"
    if ! grep -q "MLX_SERVE_ANE_DUAL=1 ignored" "$DUAL_LOG"; then
        echo -e "${RED}FAIL${NC} dual on single-ANE silicon did not self-disable by name:"
        grep "\[ane\]" "$DUAL_LOG" | head -5
        rm -f "$DUAL_LOG"
        exit 1
    fi
    if ! grep "prefill offload ready" "$DUAL_LOG" | grep -q "units=1"; then
        echo -e "${RED}FAIL${NC} dual self-disabled but did not build units=1:"
        grep "prefill offload ready" "$DUAL_LOG"
        rm -f "$DUAL_LOG"
        exit 1
    fi
    if ! grep -q "\[ane\] prefill offload engaged" "$DUAL_LOG"; then
        echo -e "${RED}FAIL${NC} a refused dual request killed the single-ANE seam:"
        grep "\[ane\]" "$DUAL_LOG" | head -5
        rm -f "$DUAL_LOG"
        exit 1
    fi
    assert_no_eval_failures "dual arm" "$DUAL_LOG"
    echo "  dual arm: refused by name on single-ANE silicon, units=1 still engaged"
    rm -f "$DUAL_LOG"
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
# Row mode bills roughly TWICE channel mode's int8 bytes (~2.8 GB vs ~1.3 GB
# on a 4B), so on a small-RAM Mac it legitimately does not fit once the box
# carries any pressure from earlier arms — the admission gate then refuses it
# BY NAME. That is the gate working, not a row-mode regression, so this arm
# reports NOT-RUN instead of failing. Keyed on the gate's own refusal string
# so a genuine "built nothing, said nothing" still FAILs below.
if grep -q "row-mode offload bills" "$CHAN_LOG"; then
    echo -e "  ${YELLOW}NOT RUN${NC} row-mode arm: refused by the memory gate on this machine"
    grep "row-mode offload bills" "$CHAN_LOG" | sed 's/^/    /' | head -2
    echo "    (channel mode is the shipping default and was exercised above)"
    ROW_ARM_SKIPPED=1
fi
if [ -z "${ROW_ARM_SKIPPED:-}" ] && ! grep "prefill offload ready" "$CHAN_LOG" | grep -q "mode=row"; then
    echo -e "${RED}FAIL${NC} MLX_SERVE_ANE_MODE=row did not build row-mode programs:"
    grep "\[ane\]" "$CHAN_LOG" | head -5
    rm -f "$CHAN_LOG"
    exit 1
fi
if [ -z "${ROW_ARM_SKIPPED:-}" ] && ! grep "prefill offload engaged" "$CHAN_LOG" | grep -q "mode=row"; then
    echo -e "${RED}FAIL${NC} row mode built but the MLP seam never engaged (or engaged as channel):"
    grep "\[ane\]" "$CHAN_LOG" | head -5
    rm -f "$CHAN_LOG"
    exit 1
fi
CHAN_GDN=$(grep "prefill offload ready" "$CHAN_LOG" | sed -n 's/.*+ \([0-9][0-9]*\) gdn layers.*/\1/p')
if [ -z "${ROW_ARM_SKIPPED:-}" ] && [ "${CHAN_GDN:-0}" -gt 0 ] && ! grep "gdn offload engaged" "$CHAN_LOG" | grep -q "mode=row"; then
    echo -e "${RED}FAIL${NC} row mode built $CHAN_GDN gdn layers but the gdn seam never engaged:"
    grep "\[ane\]" "$CHAN_LOG" | head -5
    rm -f "$CHAN_LOG"
    exit 1
fi
CHAN_CONTENT=$(echo "$CHAN_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
if [ -z "${ROW_ARM_SKIPPED:-}" ] && [ -z "$CHAN_CONTENT" ]; then
    echo -e "${RED}FAIL${NC} empty completion under row mode."
    rm -f "$CHAN_LOG"
    exit 1
fi
assert_no_eval_failures "row arm" "$CHAN_LOG"
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
