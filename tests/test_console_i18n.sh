#!/bin/bash
# Integration tests for the console's localization (GET /):
#   * the language boot is injected BEFORE the stylesheet, so the first paint is
#     already in the resolved language;
#   * the served page carries the boot and the switch;
# The boot's behaviour (positional %@, the browser's language, a stored choice
# winning, the four markup slots) is pinned in tests/html_console_test.mjs,
# which evaluates the shipped i18n.js.
#   * every key the markup marks (`index.html`, and metrics' own panel markup)
#     has a zh-Hans entry — the completeness check that actually matters;
#   * the API panel is translated for the reader while the English source the
#     MODEL is fed (`collectApi` → `systemPrompt`) is still English.
#
# Usage: ./tests/test_console_i18n.sh [port]
#   Starts its own server (no model is needed: the console is served headless).
#   BINARY overrides the server binary, e.g.
#   BINARY="/Applications/MLX Core.app/Contents/MacOS/mlx-serve"

set -u

PORT="${1:-11293}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_console_i18n.log
PAGE=/tmp/test_console_i18n_page.html
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

EMPTY_DIR="$(mktemp -d)"
trap 'kill $SERVER_PID 2>/dev/null; rm -rf "$EMPTY_DIR"' EXIT

"$BINARY" --serve --port "$PORT" --host 127.0.0.1 --model-dir "$EMPTY_DIR" --metrics --log-level warn >"$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
    curl -sf -o /dev/null "$BASE/" && break
    sleep 0.5
done

curl -sf "$BASE/" -o "$PAGE" || { echo "FAIL: GET / failed (see $LOG)"; exit 1; }

has() { grep -q -e "$1" "$PAGE" && echo 1 || echo 0; }

# Offsets, not just presence: the order is the whole point of a head boot.
check "the language boot precedes the stylesheet" \
    "$(python3 - "$PAGE" <<'PY'
import sys
page = open(sys.argv[1], encoding='utf-8', errors='replace').read()
boot = page.find('Localization for the built-in console')
style = page.find('<style>')
print(1 if 0 <= boot < style else 0)
PY
)"

check "the switch is in the sidebar head" "$(has 'id=lang-toggle')"
echo
echo "── completeness: every marked key has a zh-Hans entry ──"
MARKED=$(python3 - "$PAGE" <<'PY'
import sys, re, html
page = open(sys.argv[1], encoding='utf-8', errors='replace').read()

# The table itself, so a key defined nowhere is reported rather than skipped.
block = re.search(r'var ZH = \{(.*?)\n  \}', page, re.S)
if not block:
    print('NO_TABLE')
    sys.exit(0)
keys = set(m.group(1) for m in re.finditer(r'^\s*"((?:[^"\\]|\\.)*)":', block.group(1), re.M))
marked = [html.unescape(m) for m in re.findall(r'data-i18n(?:-title|-aria-label|-placeholder)?="([^"]*)"', page)]
missing = sorted(set(k for k in marked if k not in keys))
print('NO_TABLE' if not keys else '')
print('marked=%d distinct=%d zhkeys=%d' % (len(marked), len(set(marked)), len(keys)))
for k in missing:
    print('MISSING: ' + k)
PY
)
echo "$MARKED" | grep -v '^$'
check "the scan found the table and a real number of marked keys" \
    "$(echo "$MARKED" | grep -q '^marked=[0-9]' && echo 1 || echo 0)"
check "no marked key is missing from the zh-Hans table" \
    "$(echo "$MARKED" | grep -q 'MISSING:' && echo 0 || echo 1)"

echo
echo "── the model still reads English ──"
MODEL=$(python3 - "$PAGE" <<'PY'
import sys, re, html
page = open(sys.argv[1], encoding='utf-8', errors='replace').read()
rows = re.findall(r'<div class=d data-i18n="([^"]*)">(.*?)</div>', page, re.S)
bad = []
for key, inner in rows:
    if html.unescape(re.sub(r'<[^>]*>', '', inner)) != html.unescape(key):
        bad.append(key)
if not rows:
    print('NO_ROWS')
cjk = [k for k, _ in rows if re.search(r'[\u3400-\u9fff]', k)]
print('rows=%d' % len(rows))
for k in bad[:5]:
    print('TEXT_MISMATCH: ' + k)
for k in cjk[:5]:
    print('NOT_ENGLISH: ' + k)
PY
)
echo "$MODEL" | grep -v '^$'
check "every API row carries its own English text as the key" \
    "$(echo "$MODEL" | grep -q '^rows=[0-9]' && ! echo "$MODEL" | grep -q 'TEXT_MISMATCH' && echo 1 || echo 0)"
check "no API row's English source contains Chinese" \
    "$(echo "$MODEL" | grep -q 'NOT_ENGLISH' && echo 0 || echo 1)"
check "collectApi reads that source, not the rendered text" \
    "$(python3 - "$PAGE" <<'PY'
import sys, re
page = open(sys.argv[1], encoding='utf-8', errors='replace').read()
m = re.search(r'function collectApi.*?\n  \}\)\(\);', page, re.S)
print(1 if m and "getAttribute('data-i18n')" in m.group(0) else 0)
PY
)"

echo
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
