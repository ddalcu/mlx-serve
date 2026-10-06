#!/bin/bash
# Guard the served console's theme boot before first paint; no model required.
# English is the first-release language; Simplified Chinese remains deferred.
# Stored/OS theme choices and blocked storage are exercised by html_console_test.mjs.
# Usage: ./tests/test_console_theme.sh [port] (BINARY overrides the test binary)

set -u

PORT="${1:-11292}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG="$(mktemp)"
PAGE="$(mktemp)"
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

WORK_DIR="$(mktemp -d)"
EMPTY_DIR="$WORK_DIR/models"
mkdir -p "$EMPTY_DIR" "$WORK_DIR/home"
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; fi
    rm -rf "$WORK_DIR" "$LOG" "$PAGE"
}
trap cleanup EXIT
if [ ! -x "$BINARY" ]; then echo "FAIL: build the test binary first: $BINARY"; exit 1; fi
if curl -s --max-time 1 "$BASE/health" >/dev/null; then echo "FAIL: test port $PORT is occupied"; exit 1; fi
HOME="$WORK_DIR/home" "$BINARY" --serve --port "$PORT" --host 127.0.0.1 --model-dir "$EMPTY_DIR" --metrics --log-level warn >"$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 60); do
    kill -0 "$SERVER_PID" 2>/dev/null || { cat "$LOG"; exit 1; }
    curl -sf -o /dev/null "$BASE/" && break
    sleep 0.5
done
curl -sf "$BASE/" -o "$PAGE" || { echo "FAIL: GET / failed"; cat "$LOG"; exit 1; }
check "theme boot is in the head before the stylesheet" \
    "$(python3 - "$PAGE" <<'PYBOOT'
import sys
page = open(sys.argv[1]).read()
boot = page.find('document.documentElement.dataset.theme =')
print(int(0 <= boot < page.find('<style>') < page.find('</head>')))
PYBOOT
)"
check "System, Light and Dark choices are wired in Settings" \
    "$(node --test tests/html_console_test.mjs >/dev/null 2>&1 && grep -q 'data-pref=' "$PAGE" && echo 1 || echo 0)"
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
