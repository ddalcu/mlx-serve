#!/bin/bash
# test_stream_keepalive.sh — a streaming surface that BUFFERS (a long tool call) never leaves the socket silent past
# the keepalive interval, on chat, messages and responses: max inter-chunk gap < MAX_GAP_S, at least one
# `: keepalive` / `event: ping`, and the tool call still parses with valid-JSON arguments (keepalive bytes must not
# corrupt the SSE). A surface that finished faster than MIN_BUFFER_S SKIPs rather than passing vacuously.
#
#   KEEPALIVE_TEST_MODEL=<dir> ./tests/test_stream_keepalive.sh [port]   # default gemma-4-e4b-it-4bit
#
# Hermetic twin: StreamHeartbeat tests. Story: "A buffered streaming surface must beat on SOCKET SILENCE" (server-http).

set -e
source "$(dirname "$0")/lib/server.sh"

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

srv_wait_health "http://127.0.0.1:$PORT" 90 1 || true
if ! curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
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
