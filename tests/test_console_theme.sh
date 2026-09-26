#!/bin/bash
# Integration tests for the console's theme (GET /):
#   * the theme boot is injected BEFORE the stylesheet, so a light page never
#     paints dark first;
#   * the served page carries the switch.
# The boot's behaviour (OS default, a stored choice, toggling, a store that
# throws) and the palettes' variable parity are pinned in
# tests/html_console_test.mjs, which evaluates the shipped files.
#
# Usage: ./tests/test_console_theme.sh [port]
#   Starts its own server (no model is needed: the console is served headless).
#   BINARY overrides the server binary, e.g. BINARY="/Applications/MLX Core.app/Contents/MacOS/mlx-serve"

set -u

PORT="${1:-11292}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_console_theme.log
PAGE=/tmp/test_console_theme_page.html
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
    echo "SKIP: server binary not found: $BINARY (build with: zig build -Doptimize=ReleaseFast)"
    exit 0
fi

"$BINARY" --serve --port "$PORT" --host 127.0.0.1 --model-dir /tmp --metrics --log-level warn >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for _ in $(seq 1 60); do
    curl -sf -o /dev/null "$BASE/" && break
    sleep 0.5
done

curl -sf "$BASE/" -o "$PAGE" || { echo "FAIL: GET / failed (see $LOG)"; exit 1; }

has() { grep -q -e "$1" "$PAGE" && echo 1 || echo 0; }

# Offsets, not just presence: the order is the whole point of the boot file.
check "the theme boot precedes the stylesheet" \
    "$(python3 - "$PAGE" <<'PY'
import sys
page = open(sys.argv[1], encoding='utf-8', errors='replace').read()
boot = page.find('Theme boot for the built-in console')
style = page.find('<style>')
print(1 if 0 <= boot < style else 0)
PY
)"

check "the switch is in the sidebar head" "$(has 'id=theme-toggle')"

echo
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
