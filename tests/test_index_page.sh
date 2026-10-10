#!/bin/bash
# Guard: the built-in console at `GET /` is served on a server with NO model, byte for byte.
#
# `GET /` once rendered one *LoadedModel, so a headless boot answered 503 at the
# root, the first page a person opens. The page is the model PICKER, so it has to
# render before anything is loaded. It is also built from app-web/ and embedded
# as is: the bytes on the wire must be src/html/index.html, nothing templated.
# What the page does (panes, Markdown, i18n, Monitoring) is app-web's own test
# suite (`cd app-web && npm test`); the metrics probe the page keys off is here.
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

echo "[1/2] headless GET /"
if boot; then
    STATUS=$(curl -s -o "$BODY" -w '%{http_code}' "http://127.0.0.1:$PORT/")
    CT=$(curl -s -D - -o /dev/null "http://127.0.0.1:$PORT/" | grep -i '^content-type:' | tr -d '\r')
    check "GET / with no model loaded → 200 (got $STATUS)" \
        "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"
    check "Content-Type is text/html" \
        "$(echo "$CT" | grep -qi 'text/html' && echo 1 || echo 0)"
    check "the bytes served are src/html/index.html, untouched" \
        "$(cmp -s "$BODY" src/html/index.html && echo 1 || echo 0)"
    check "it is the console page (one #app mount)" \
        "$(grep -q '<div id="app">' "$BODY" && echo 1 || echo 0)"

    # The console reads "metrics off" from a 503 here, so the status code is its contract.
    METRICS_STATUS=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/metrics.json")
    check "GET /metrics.json without --metrics → 503 (got $METRICS_STATUS)" \
        "$([ "$METRICS_STATUS" = "503" ] && echo 1 || echo 0)"
    stop
else
    check "headless boot" 0
fi

echo "[2/2] --metrics"
if boot --metrics; then
    STATUS=$(curl -s -o "$BODY" -w '%{http_code}' "http://127.0.0.1:$PORT/")
    METRICS_STATUS=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/metrics.json")
    check "page is the same file with --metrics (got $STATUS)" \
        "$([ "$STATUS" = "200" ] && cmp -s "$BODY" src/html/index.html && echo 1 || echo 0)"
    check "GET /metrics.json with --metrics → 200 (got $METRICS_STATUS)" \
        "$([ "$METRICS_STATUS" = "200" ] && echo 1 || echo 0)"
    stop
else
    check "boot with --metrics" 0
fi

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
