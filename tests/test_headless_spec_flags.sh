#!/bin/bash
# test_headless_spec_flags.sh — headless serve (`--serve --model-dir`, the mode the app always launches) honours the
# CLI's spec flags: the boot banner echoes non-default --pld-draft-len / --pld-key-len, --no-pld still wins, a bare
# boot reports 5/3. [4/4] endpoint existence never depends on model state: an unknown path is a 404 while served
# endpoints answer 503 with no default model (llmprobe read a catch-all 503 as every surface absent).
#
#   ./tests/test_headless_spec_flags.sh [port]
#
# FULLY HERMETIC: an empty --model-dir discovers nothing and loads nothing; runs in seconds. Hermetic twin: PldDefaults.

set -u
source "$(dirname "$0")/lib/server.sh"

PORT="${1:-11265}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

EMPTY_DIR="$(mktemp -d)"
LOG="$(mktemp)"
cleanup() {
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    rm -rf "$EMPTY_DIR" "$LOG"
}
trap cleanup EXIT

# Boot headless over the empty dir, capture the banner, stop. Prints nothing;
# the caller greps "$LOG".
boot() {
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    sleep 0.5
    : > "$LOG"
    "$BINARY" --serve --model-dir "$EMPTY_DIR" --port "$PORT" --log-file off "$@" > "$LOG" 2>&1 &
    local pid=$!
    srv_wait_health "http://127.0.0.1:$PORT" 60 0.5 "$pid" || true
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
        echo "  (server did not come up; log follows)"; cat "$LOG"; kill "$pid" 2>/dev/null; return 1
    }
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    return 0
}

echo "Headless spec-decode flag plumbing (port $PORT)"

echo "[1/5] --pld with non-default draft/key lengths"
if boot --pld --pld-draft-len 8 --pld-key-len 4; then
    grep -q "PLD speculative decoding: ENABLED" "$LOG"
    check "--pld enables PLD in headless mode" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    grep -q "draft_len=8" "$LOG"
    check "--pld-draft-len 8 reaches the request defaults" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    grep -q "key_len=4" "$LOG"
    check "--pld-key-len 4 reaches the request defaults" "$([ $? -eq 0 ] && echo 1 || echo 0)"
else
    check "boot with --pld" 0
fi

echo "[2/5] --no-pld still disables"
if boot --no-pld --pld-draft-len 8 --pld-key-len 4; then
    grep -q "PLD speculative decoding: ENABLED" "$LOG"
    check "--no-pld keeps PLD off even with lengths passed" "$([ $? -ne 0 ] && echo 1 || echo 0)"
else
    check "boot with --no-pld" 0
fi

echo "[3/5] bare default matches the documented 5/3"
if boot; then
    grep -q "draft_len=5, key_len=3" "$LOG"
    check "default headless boot is PLD on at 5/3" "$([ $? -eq 0 ] && echo 1 || echo 0)"
else
    check "bare default boot" 0
fi

# `--max-tokens` used to feed only the offline `--prompt` mode; headless hand-rolls its own
# ServerConfig literal, which is how the --pld* trio was eaten.
echo "[4/5] --max-tokens reaches the serve-mode omitted-field default"
if boot --max-tokens 4096; then
    grep -q "default max_tokens for omitted requests: 4096" "$LOG"
    check "--max-tokens 4096 reaches the headless request defaults" "$([ $? -eq 0 ] && echo 1 || echo 0)"
else
    check "boot with --max-tokens" 0
fi
if boot; then
    grep -q "default max_tokens for omitted requests" "$LOG"
    check "no flag announces no default (offline's 100 never leaks into serve)" "$([ $? -ne 0 ] && echo 1 || echo 0)"
else
    check "bare boot for the --max-tokens default" 0
fi

# Model resolution runs BEFORE dispatch, so with no default model an unknown
# path never reached the chain's 404 and came back 503 "No default model
# configured". A path's existence has nothing to do with model state, and a
# server that answers every unknown path with a non-404 looks like a catch-all
# to anything that maps endpoints by probing them — llmprobe scored every
# surface absent against a headless boot for exactly this reason (2026-07-25).
# Headless with an EMPTY model dir is the only place this is observable.
echo "[5/5] endpoint existence does not depend on a model being loaded"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 0.5
: > "$LOG"
"$BINARY" --serve --model-dir "$EMPTY_DIR" --port "$PORT" --log-file off > "$LOG" 2>&1 &
HPID=$!
srv_wait_health "http://127.0.0.1:$PORT" 60 0.5 "$HPID" || true
post_code() {
    curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT$1" \
        -H 'Content-Type: application/json' -d '{}'
}
unknown_code=$(post_code /v1/__no_such_endpoint__)
known_code=$(post_code /v1/chat/completions)
edits_code=$(post_code /v1/images/edits)
kill "$HPID" 2>/dev/null
wait "$HPID" 2>/dev/null
check "unknown endpoint is 404, not no-model (got $unknown_code)" \
    "$([ "$unknown_code" = "404" ] && echo 1 || echo 0)"
check "a served endpoint still reports 503 no-model (got $known_code)" \
    "$([ "$known_code" = "503" ] && echo 1 || echo 0)"
check "a served media endpoint is not mistaken for absent (got $edits_code)" \
    "$([ "$edits_code" = "503" ] && echo 1 || echo 0)"

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
