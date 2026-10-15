#!/bin/bash
# Streaming-silence regression test (class guard).
#
# CLASS: a streaming surface that BUFFERS tokens must never let the socket go
# silent for longer than the keepalive interval.
#
# The server buffers generated tokens whenever it might be looking at a tool
# call (`chat.streamShouldBufferForTools`) or an unclosed thinking block
# (`chat.streamThinkGate` -> .hold_thinking). For a large tool call — an agent
# one-shotting a whole file into `write_file` — that buffer holds for the
# ENTIRE generation, and pre-fix the server wrote zero bytes for its whole
# duration. Clients with an idle-body timeout tear such a connection down
# mid-generation: Node's `fetch` (undici) defaults to `bodyTimeout: 300_000`,
# so any tool call that takes >5 min to generate dies as `TypeError: terminated`.
# Live failure 2026-07-08: a pi agent session building a JS game lost two
# ~5-minute `write` calls exactly this way (~10 min of GPU discarded).
#
# The keepalive used to fire only on `.idle` — i.e. only while WAITING for the
# first token (long prefill). Once tokens flowed into a buffer, it never fired.
#
# This test asserts, for EVERY streaming surface (chat / messages / responses):
#   1. the max inter-chunk gap stays under MAX_GAP_S even though the model
#      spends far longer than that inside one buffered tool call
#   2. at least one keepalive (`: keepalive` comment / `event: ping`) arrives
#   3. the tool call still parses — name intact, arguments are valid JSON
#      (the injected keepalive bytes must not corrupt the SSE stream)
#
# A surface whose generation finishes faster than MIN_BUFFER_S never exercised
# the buffer, so it SKIPs rather than passing vacuously.
#
# Requires:
#   - A built mlx-serve binary (zig build -Doptimize=ReleaseFast)
#   - KEEPALIVE_TEST_MODEL or ~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit
#
# Usage: ./tests/test_stream_keepalive.sh [port]

set -e

PORT=${1:-11296}
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

source "$(dirname "$0")/_lib_models.sh"
MODEL="${KEEPALIVE_TEST_MODEL:-$(find_model mlx-community/gemma-4-e4b-it-4bit)}"
if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_stream_keepalive: model directory not found ($MODEL)."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

# Keepalive interval is 5s (Conn.STREAM_KEEPALIVE_MS). Allow generous slack for
# a busy GPU: what we are proving is "seconds, not minutes".
MAX_GAP_S=${MAX_GAP_S:-15}
# A generation shorter than this never filled the buffer long enough to prove
# anything, so the surface SKIPs instead of passing vacuously.
MIN_BUFFER_S=${MIN_BUFFER_S:-8}

LOG=$(mktemp -t keepalive-server.XXXXXX)
# The probe generates ~2.4k tokens; a pinned context keeps an auto-sized one
# (small when free memory is low at boot) from cutting the call short.
"$BINARY" --model "$MODEL" --serve --port "$PORT" --host 127.0.0.1 --log-level info --ctx-size 8192 >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; rm -f "$LOG"; }
trap cleanup EXIT

for _ in $(seq 1 90); do
    sleep 1
    curl -s -m 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
done
if ! curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo -e "${RED}FAIL${NC} server did not come up"; tail -20 "$LOG"; exit 1
fi

FAILURES=0
for SURFACE in chat messages responses; do
    echo "── surface: $SURFACE ──"
    set +e
    OUT=$(MAX_GAP_S="$MAX_GAP_S" MIN_BUFFER_S="$MIN_BUFFER_S" \
        python3 "$(dirname "$0")/stream_keepalive_probe.py" "$PORT" "$SURFACE" "$(basename "$MODEL")")
    RC=$?
    set -e
    echo "$OUT" | sed 's/^/  /'
    case $RC in
        0) echo -e "  ${GREEN}PASS${NC} $SURFACE" ;;
        2) echo -e "  ${YELLOW}SKIP${NC} $SURFACE (generation too short to exercise the buffer)" ;;
        *) echo -e "  ${RED}FAIL${NC} $SURFACE"; FAILURES=$((FAILURES + 1)) ;;
    esac
done

echo
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}ALL PASS${NC} — no streaming surface goes silent during a buffered tool call"
    exit 0
fi
echo -e "${RED}$FAILURES surface(s) FAILED${NC}"
exit 1
