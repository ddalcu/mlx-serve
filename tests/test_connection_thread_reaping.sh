#!/bin/bash
# Regression: one completed HTTP request must not retain one pthread stack.
#
# The accept loop used to discard each joinable std.Thread handle. The handler
# returned, but macOS kept its stack mapping and page-table bookkeeping until a
# join/detach that never came. A long Claude Code session therefore grew from
# the model's 4-5 GB working set toward 10 GB while MLX cache_bytes stayed tiny.

set -u

PORT="${1:-11483}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
REQUESTS="${CONN_REAP_REQUESTS:-300}"
EMPTY_DIR="$(mktemp -d)"
LOG="$(mktemp)"
SERVER_PID=""

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null
    rm -rf "$EMPTY_DIR" "$LOG"
}
trap cleanup EXIT

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

"$BINARY" --serve --model-dir "$EMPTY_DIR" --host 127.0.0.1 \
    --port "$PORT" --log-file off >"$LOG" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
        READY=1
        break
    fi
    sleep 0.1
done
if [ "$READY" -ne 1 ]; then
    echo "[fail] server did not start"
    cat "$LOG"
    exit 1
fi

sleep 0.5
# macOS counts vmmap's Stack regions. Linux merges adjacent mappings, so it reads the mapped
# size in MiB: a retained stack is 16 MiB there, and glibc's malloc arenas move it by 64.
if [ "$(uname)" = Darwin ]; then UNIT="stack regions"; LIMIT=8; else UNIT="MiB mapped"; LIMIT=256; fi
stack_regions() {
    if [ "$(uname)" = Darwin ]; then
        vmmap "$SERVER_PID" 2>/dev/null | grep -Ec '^Stack[[:space:]]+[[:xdigit:]]+-'
    else
        awk '/^VmSize:/ {print int($2 / 1024)}' "/proc/$SERVER_PID/status" 2>/dev/null || echo 0
    fi
}

BEFORE="$(stack_regions)"
if [ "$BEFORE" -lt 1 ]; then
    echo "[fail] no $UNIT for pid $SERVER_PID — nothing was measured"
    exit 1
fi
for _ in $(seq 1 "$REQUESTS"); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || {
        echo "[fail] health request failed"
        exit 1
    }
done
sleep 1
AFTER="$(stack_regions)"
if [ "$AFTER" -lt 1 ]; then
    echo "[fail] no $UNIT after the requests — server died or the probe failed"
    exit 1
fi
DELTA=$((AFTER - BEFORE))

echo "connection thread reaping: before=$BEFORE after=$AFTER delta=$DELTA requests=$REQUESTS"
if [ "$DELTA" -gt "$LIMIT" ]; then
    echo "[fail] completed requests retained $DELTA $UNIT"
    exit 1
fi

curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || {
    echo "[fail] server stopped responding"
    exit 1
}
echo "[pass] connection thread stacks are reaped"
