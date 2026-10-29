#!/bin/bash
# test_dspark_lfm2.sh — DSpark (LiquidAI LFM2.5, 2.6B dense hybrid and 8B-A1B lfm2_moe) on a target whose drafter/
# holds the sidecar. [1] the sidecar is CLASSIFIED as DSpark (`dspark: markov head rank=`) and the hybrid trunk no
# longer vetoes it (`[spec-wiring] ... dflash=true`); [2] rounds ENGAGE with accepts > 0; [3] greedy bytes equal a
# --no-drafter boot on the RAW /v1/completions surface (a chat model's reasoning amplifies a near-tie flip), on an echo
# prompt so partial-accept rounds (the conv-state rollback) still happen.
#
#   DSPARK_TEST_MODEL=<target dir> ./tests/test_dspark_lfm2.sh   # DSPARK_TEST_PORT (11357); SKIPs without drafter/
#
# Hermetic twins: the split-contract, anchor_row_drafts and markov-head parse tests, dflashGateMinimum.

set -euo pipefail

MODEL="${DSPARK_TEST_MODEL:-}"
if [ -z "$MODEL" ]; then echo "SKIP: DSPARK_TEST_MODEL not set"; exit 0; fi
if [ ! -f "$MODEL/drafter/config.json" ]; then
    echo "SKIP: $MODEL/drafter/config.json not found"; exit 0
fi

PORT="${DSPARK_TEST_PORT:-11357}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

boot() { # $1 = log file, $2... = extra args
    local log="$1"; shift
    cleanup
    "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --ctx-size 8192 --prefix-cache-entries 0 --log-level debug "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do curl -s -m 2 "$BASE/health" > /dev/null 2>&1 && return 0; sleep 1; done
    echo "FAIL: server did not come up"; cat "$log"; exit 1
}

# An ECHO prompt on the RAW completions surface: no chat template, so no
# reasoning block to amplify one near-tie argmax flip into two different
# plans, and the repetition keeps acceptance high so partial-accept rounds
# (the conv-state rollback) actually happen.
ECHO_TEXT=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog number %d.' % i for i in range(40)]))")
BODY=$(python3 - "$ECHO_TEXT" <<'PY'
import json, sys
print(json.dumps({"model": "m", "max_tokens": 300, "temperature": 0, "prompt": sys.argv[1]}))
PY
)

ask() { curl -s -m 300 "$BASE/v1/completions" -H 'content-type: application/json' -d "$BODY" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])"; }

L1=$(mktemp /tmp/dspark_serial.XXXXXX); L2=$(mktemp /tmp/dspark_on.XXXXXX);

echo "[1] serial reference"
boot "$L1" --no-drafter
SERIAL=$(ask)

echo "[2] DSpark engaged"
boot "$L2"
grep -q "dspark: markov head rank=" "$L2" || { echo "FAIL: sidecar not classified as DSpark"; exit 1; }
DS=$(ask)
grep -q "\[spec-wiring\].*dflash=true" "$L2" || { echo "FAIL: dflash not wired (hybrid veto?)"; exit 1; }
STATS=$(grep -o "mode=dflash.*per_draft_pct=[0-9.]*%" "$L2" | tail -1)
[ -n "$STATS" ] || { echo "FAIL: no dflash rounds"; exit 1; }
echo "    $STATS"
ACC=$(echo "$STATS" | sed -n 's/.*accepts=\([0-9]*\).*/\1/p')
[ "${ACC:-0}" -gt 0 ] || { echo "FAIL: zero accepted drafts"; exit 1; }

echo "[3] greedy output identical to serial"
[ "$SERIAL" = "$DS" ] || { echo "FAIL: DSpark greedy output differs from serial"; diff <(echo "$SERIAL") <(echo "$DS") | head; exit 1; }

rm -f "$L1" "$L2"
echo "PASS: DSpark on LFM2.5"
