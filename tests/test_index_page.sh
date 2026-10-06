#!/bin/bash
# Guard: the built-in console at `GET /` must render on a server with NO model.
#
# `GET /` was dispatched AFTER model resolution and rendered one *LoadedModel,
# so a headless boot — `mlx-serve serve` / `--serve --model-dir`, now the
# default way the server starts and the only way the app launches it — answered
# 503 {"error":"No default model configured"} at the root. The page is the
# thing a person opens first, and it is also the model PICKER, so it has to
# render before anything is loaded, by construction.
#
# Also pins the two properties a page rewrite can silently drop:
#   * every endpoint the server serves is documented (the reference had been
#     missing the whole Ollama /api/* surface);
#   * the live-metrics mount is present with --metrics and absent without it
#     (deliberately duplicates one test_metrics.sh assertion — that script
#     needs a real checkpoint, this one doesn't).
#
# FULLY HERMETIC: an empty --model-dir discovers zero models, so no weights are
# needed and the whole thing runs in seconds (same trick as
# tests/test_headless_spec_flags.sh).
#
# Usage: ./tests/test_index_page.sh [port]

set -u

PORT="${1:-11266}"
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

WORK_DIR="$(mktemp -d)"
EMPTY_DIR="$WORK_DIR/models"
mkdir -p "$EMPTY_DIR" "$WORK_DIR/home"
LOG="$(mktemp)"
BODY="$(mktemp)"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null
    rm -rf "$WORK_DIR" "$LOG" "$BODY"
}
trap cleanup EXIT

boot() {
    : > "$LOG"
    HOME="$WORK_DIR/home" "$BINARY" --serve --host 127.0.0.1 --model-dir "$EMPTY_DIR" --port "$PORT" --log-file off "$@" > "$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 60); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 0.5
        kill -0 "$SERVER_PID" 2>/dev/null || break
    done
    echo "  (server did not come up; log follows)"; cat "$LOG"
    return 1
}

stop() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}

if curl -s --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "FAIL: test port $PORT is occupied"; exit 1
fi

echo "Built-in console at GET / (port $PORT, no model)"

# ── 1. It renders at all without a model ────────────────────────────────────
echo "[1/3] headless GET /"
if boot; then
    STATUS=$(curl -s -o "$BODY" -w '%{http_code}' "http://127.0.0.1:$PORT/")
    CT=$(curl -s -D - -o /dev/null "http://127.0.0.1:$PORT/" | grep -i '^content-type:' | tr -d '\r')
    check "GET / with no model loaded → 200 (got $STATUS)" \
        "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"
    check "Content-Type is text/html" \
        "$(echo "$CT" | grep -qi 'text/html' && echo 1 || echo 0)"
    check "no 'No default model configured' in the body" \
        "$(grep -q 'No default model configured' "$BODY" && echo 0 || echo 1)"

    # ── 2. The console + the full endpoint reference are in the page ────────
    echo "[2/3] console markup + endpoint coverage"
    for pane in models monitoring api settings image video audio library; do
        check "navigation destination '$pane' bundled" \
            "$(grep -q "nav(\"$pane\"" "$BODY" && echo 1 || echo 0)"
    done
    for id in app content session-list chat-input chat-send image-files monitor-sessions; do
        check "console mount/control '$id' present" \
            "$(grep -q "id=\"$id\"" "$BODY" && echo 1 || echo 0)"
    done
    check "Chat is the initial view" \
        "$(grep -q 'view = "chat"' "$BODY" && echo 1 || echo 0)"
    check "unified Monitoring range picker present" \
        "$(grep -q 'data-range' "$BODY" && echo 1 || echo 0)"
    # Read the route table, then check the documentation, not route strings in JS.
    check "every served endpoint has a documentation row" \
        "$(python3 - "$BODY" <<'PYROUTES'
import re, sys
page = open(sys.argv[1]).read()
source = open('src/server.zig').read()
block = re.search(r'const ROUTE_PATHS = .*?\{(.*?)\n\};', source, re.S)
routes = re.findall(r'"(/[^" ]*)"', block.group(1)) if block else []
rows = set(re.findall(r'<td>(/[^<]*)</td>', page))
print(int(len(routes) > 30 and all(path in rows for path in routes)))
PYROUTES
)"

    check "no metrics panel mount without --metrics" \
        "$(grep -q 'id=mlx-metrics' "$BODY" && echo 0 || echo 1)"
    stop
else
    check "headless boot" 0
fi

# ── 3. --metrics puts the live panel in the header ──────────────────────────
echo "[3/3] --metrics panel mount"
if boot --metrics; then
    STATUS=$(curl -s -o "$BODY" -w '%{http_code}' "http://127.0.0.1:$PORT/")
    check "page still 200 with --metrics (got $STATUS)" \
        "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"
    check "metrics mount present with --metrics" \
        "$(grep -q 'id=mlx-metrics' "$BODY" && echo 1 || echo 0)"
    check "metrics-enabled boot injected" \
        "$(grep -q 'dataset.studioMetrics = "enabled"' "$BODY" && echo 1 || echo 0)"
    stop
else
    check "boot with --metrics" 0
fi

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
