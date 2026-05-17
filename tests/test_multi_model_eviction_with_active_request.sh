#!/bin/bash
# Plan 05 Phase F — eviction is blocked by in-flight requests.
#
# Server: max-resident-models 2 with 3 models in --model-dir. Start a long
# streaming request on M1, then fire a request on M3 (which would normally
# evict the LRU of {M1, M2}). Because M1 has an in-flight refcount, the
# evictor must pick M2 instead (or wait). M1's stream must complete
# unaffected; M3's request also completes.
#
# Usage: ./tests/test_multi_model_eviction_with_active_request.sh [root] [port]

set -e

ROOT="${1:-$HOME/.mlx-serve/models}"
PORT="${2:-8099}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$ROOT" ]; then
    echo -e "${YELLOW}SKIP${NC} eviction-with-active: $ROOT not found."
    exit 0
fi
source "$(dirname "$0")/_lib_supported_models.sh"
MODELS=()
while IFS= read -r m; do MODELS+=("$m"); done < <(list_supported_models "$ROOT" 3)
if [ "${#MODELS[@]}" -lt 3 ]; then
    echo -e "${YELLOW}SKIP${NC} eviction-with-active: need >=3 models."
    exit 0
fi
M1="${MODELS[0]}"; M2="${MODELS[1]}"; M3="${MODELS[2]}"
echo "  using $M1 (active), $M2 (lru-victim), $M3 (incoming)"

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
LOGFILE=$(mktemp); STREAM1=$(mktemp); RESP3=$(mktemp)
"$BINARY" --model-dir "$ROOT" --model "$ROOT/$M1" --serve --port "$PORT" \
    --max-resident-models 2 --max-concurrent 4 ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LOGFILE" "$STREAM1" "$RESP3"
}
trap cleanup EXIT

for _ in $(seq 1 30); do
    curl -fs "$BASE/health" >/dev/null 2>&1 && break
    sleep 1
done

# Warm M2 so it counts as the LRU victim.
echo "  warming M2..."
curl -fs -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M2\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":4}" >/dev/null

FAIL=0

# Start a long streaming request on M1 in the background.
echo "  starting long stream on M1..."
curl -fs --no-buffer -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M1\",\"messages\":[{\"role\":\"user\",\"content\":\"Recite the alphabet, one letter per line.\"}],\"max_tokens\":200,\"stream\":true}" \
    > "$STREAM1" 2>&1 &
STREAM_PID=$!
sleep 1  # give the stream a chance to actually be in-flight

# Fire M3 (should evict M2, not M1 since M1 has refcount > 0).
echo "  firing M3 (forces eviction)..."
HTTP_STATUS=$(curl -s -o "$RESP3" -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M3\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":4}")

# Wait for the stream to finish.
wait $STREAM_PID && S1=$? || S1=$?

if [ "$HTTP_STATUS" = "200" ]; then
    echo -e "${GREEN}PASS${NC} M3 request: HTTP 200"
else
    echo -e "${RED}FAIL${NC} M3 request: HTTP $HTTP_STATUS"
    FAIL=1
fi
if [ "$S1" -eq 0 ] && [ -s "$STREAM1" ]; then
    echo -e "${GREEN}PASS${NC} M1 stream completed without error"
else
    echo -e "${RED}FAIL${NC} M1 stream exit=$S1, size=$(wc -c < "$STREAM1")"
    FAIL=1
fi

if grep -q "evicting model id=$M2" "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} server log: M2 (idle LRU) was evicted, not M1"
elif grep -q "evicting model id=$M1" "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} server log: M1 was evicted while in-flight"
    FAIL=1
else
    echo -e "${YELLOW}WARN${NC} no eviction log line; check log:"; grep -i evict "$LOGFILE" || true
fi

exit $FAIL
