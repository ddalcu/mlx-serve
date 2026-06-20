#!/bin/bash
# Integration tests for the JSON admin API.
#
# Tests:
#  1. GET /admin/metrics.json without --metrics → 503
#  2. GET /admin/metrics.json with --metrics → 200 + valid JSON
#  3. POST /admin/evict-prefix-cache without --admin-key → 200 (open mode)
#  4. POST /admin/evict-prefix-cache with --admin-key, no auth header → 401
#  5. POST /admin/evict-prefix-cache with wrong key → 401
#  6. POST /admin/evict-prefix-cache with correct key → 200 {"ok":true}
#
# Usage: ./tests/test_admin_api.sh [model_dir] [port]

set -u

MODEL="${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit}"
PORT="${2:-11292}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_admin_api.log
PASS=0
FAIL=0
ADMIN_KEY="test-secret-key-123"

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

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL"
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

wait_health() {
    for _ in $(seq 1 90); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "FAIL: server never became healthy on port $PORT"
    return 1
}

# ── Phase 1: without --metrics, GET /admin/metrics.json returns 503 ──────────
echo ""
echo "── Phase 1: without --metrics ──"
"$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT
wait_health

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/admin/metrics.json")
check "GET /admin/metrics.json without --metrics → 503" "$([ "$STATUS" = "503" ] && echo 1 || echo 0)"

kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true; sleep 1

# ── Phase 2: with --metrics, JSON endpoint works ─────────────────────────────
echo ""
echo "── Phase 2: GET /admin/metrics.json with --metrics ──"
"$BINARY" --model "$MODEL" --serve --port "$PORT" --metrics --no-pld --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT
wait_health

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/admin/metrics.json")
check "GET /admin/metrics.json → 200" "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"

CT=$(curl -s -D - -o /dev/null "$BASE/admin/metrics.json" | grep -i "^content-type:" | tr -d '\r')
check "Content-Type is application/json" "$(echo "$CT" | grep -q "application/json" && echo 1 || echo 0)"

BODY=$(curl -s "$BASE/admin/metrics.json")
check "JSON has 'counters' key" "$(echo "$BODY" | grep -q '"counters"' && echo 1 || echo 0)"
check "JSON has 'gauges' key" "$(echo "$BODY" | grep -q '"gauges"' && echo 1 || echo 0)"
check "JSON has 'histograms' key" "$(echo "$BODY" | grep -q '"histograms"' && echo 1 || echo 0)"
check "JSON has 'time_to_first_token_seconds'" \
    "$(echo "$BODY" | grep -q '"time_to_first_token_seconds"' && echo 1 || echo 0)"
check "JSON has 'bucket_counts'" "$(echo "$BODY" | grep -q '"bucket_counts"' && echo 1 || echo 0)"

# Evict in open mode (no --admin-key)
EVICT=$(curl -s -X POST "$BASE/admin/evict-prefix-cache")
EVICT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/admin/evict-prefix-cache")
check "POST /admin/evict-prefix-cache open mode → 200" "$([ "$EVICT_STATUS" = "200" ] && echo 1 || echo 0)"
check "evict response has 'ok'" "$(echo "$EVICT" | grep -q '"ok"' && echo 1 || echo 0)"

# Dashboard (served regardless of --metrics; degrades gracefully without it)
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/admin")
check "GET /admin returns 200" "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"

CT=$(curl -s -D - -o /dev/null "$BASE/admin" | grep -i "^content-type:" | tr -d '\r')
check "GET /admin Content-Type is text/html" "$(echo "$CT" | grep -qi "text/html" && echo 1 || echo 0)"

DASH=$(curl -s "$BASE/admin")
check "Dashboard has stat panel elements" "$(echo "$DASH" | grep -q 'id="req-rate"' && echo "$DASH" | grep -q 'id="gpu-pct"' && echo 1 || echo 0)"
check "Dashboard has Grafana config section" "$(echo "$DASH" | grep -q 'prom-cfg' && echo 1 || echo 0)"

kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true; sleep 1

# ── Phase 3: with --admin-key, auth is enforced ───────────────────────────────
echo ""
echo "── Phase 3: --admin-key auth enforcement ──"
"$BINARY" --model "$MODEL" --serve --port "$PORT" --metrics --admin-key "$ADMIN_KEY" \
    --no-pld --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT
wait_health

# GET /admin/metrics.json should NOT require auth (read-only)
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/admin/metrics.json")
check "GET /admin/metrics.json requires no auth" "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"

# POST without any auth header → 401
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/admin/evict-prefix-cache")
check "POST evict without auth header → 401" "$([ "$STATUS" = "401" ] && echo 1 || echo 0)"

# POST with wrong key → 401
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Authorization: Bearer wrong-key" "$BASE/admin/evict-prefix-cache")
check "POST evict with wrong key → 401" "$([ "$STATUS" = "401" ] && echo 1 || echo 0)"

# POST with correct key → 200
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Authorization: Bearer $ADMIN_KEY" "$BASE/admin/evict-prefix-cache")
EVICT=$(curl -s -X POST -H "Authorization: Bearer $ADMIN_KEY" "$BASE/admin/evict-prefix-cache")
check "POST evict with correct key → 200" "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"
check "evict response has 'ok':true" "$(echo "$EVICT" | grep -q '"ok":true' && echo 1 || echo 0)"

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}PASS${NC} $TOTAL/$TOTAL tests passed"
    exit 0
else
    echo -e "${RED}FAIL${NC} $FAIL/$TOTAL tests failed"
    echo ""
    echo "--- Server log (last 20 lines) ---"
    tail -20 "$LOG" 2>/dev/null || true
    exit 1
fi
