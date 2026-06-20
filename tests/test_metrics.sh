#!/bin/bash
# Integration tests for the Prometheus metrics endpoint (GET /metrics).
#
# Tests:
#  1. Without --metrics: GET /metrics → 503
#  2. With    --metrics: GET /metrics → 200 + valid Prometheus text format
#  3. After one chat request: request_success_total=1, latency histograms present
#  4. Content-Type header is the correct Prometheus MIME type
#
# Usage: ./tests/test_metrics.sh [model_dir] [port]
#   Starts its own servers. Default model: Gemma 4 E4B 8-bit.

set -u

MODEL="${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit}"
PORT="${2:-11291}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_metrics.log
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

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL (set MODEL_DIR or pass as first arg)"
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

# ── Helper: wait for /health ──────────────────────────────────────────────
wait_health() {
    for _ in $(seq 1 90); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "FAIL: server never became healthy on port $PORT"
    return 1
}

# ════════════════════════════════════════════════════════════════════════════
# Test 1: Without --metrics, GET /metrics returns 503
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 1: without --metrics ──"

"$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT

wait_health

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics")
check "GET /metrics without --metrics returns 503" "$([ "$STATUS" = "503" ] && echo 1 || echo 0)"

BODY=$(curl -s "$BASE/metrics")
check "503 body mentions 'not enabled'" "$(echo "$BODY" | grep -q "not enabled" && echo 1 || echo 0)"

kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null || true
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

# ════════════════════════════════════════════════════════════════════════════
# Test 2: With --metrics, GET /metrics returns 200 + valid Prometheus text
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 2: with --metrics (idle — no requests yet) ──"

# Use --log-level info so the "Prometheus metrics: ENABLED" startup message is
# visible (log.info is suppressed at --log-level warn).
"$BINARY" --model "$MODEL" --serve --port "$PORT" --metrics --no-pld --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT

wait_health

# Check startup log mentions metrics
check "startup log: 'Prometheus metrics: ENABLED'" \
    "$(grep -q "Prometheus metrics: ENABLED" "$LOG" && echo 1 || echo 0)"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics")
check "GET /metrics with --metrics returns 200" "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"

# Content-Type must be the Prometheus text format MIME type.
# Use -D - (dump headers to stdout) + -o /dev/null (discard body) via GET —
# avoid `curl -I` (HEAD) because the server only routes GET to /metrics.
CT=$(curl -s -D - -o /dev/null "$BASE/metrics" | grep -i "^content-type:" | tr -d '\r')
check "Content-Type is Prometheus text MIME" \
    "$(echo "$CT" | grep -q "text/plain" && echo "$CT" | grep -q "version=0.0.4" && echo 1 || echo 0)"

BODY=$(curl -s "$BASE/metrics")

# Must have counter and histogram HELP/TYPE lines
check "# HELP vllm:prompt_tokens_total present" \
    "$(echo "$BODY" | grep -q "# HELP vllm:prompt_tokens_total" && echo 1 || echo 0)"
check "# TYPE vllm:prompt_tokens_total counter" \
    "$(echo "$BODY" | grep -q "# TYPE vllm:prompt_tokens_total counter" && echo 1 || echo 0)"
check "# TYPE vllm:time_to_first_token_seconds histogram" \
    "$(echo "$BODY" | grep -q "# TYPE vllm:time_to_first_token_seconds histogram" && echo 1 || echo 0)"
check "TTFT +Inf bucket present" \
    "$(echo "$BODY" | grep -q 'vllm:time_to_first_token_seconds_bucket{le="+Inf"}' && echo 1 || echo 0)"
check "vllm:num_requests_running gauge present" \
    "$(echo "$BODY" | grep -q "# TYPE vllm:num_requests_running gauge" && echo 1 || echo 0)"
check "mlx_serve:gpu_utilization_pct gauge present" \
    "$(echo "$BODY" | grep -q "mlx_serve:gpu_utilization_pct" && echo 1 || echo 0)"
check "mlx_serve:memory_mb gauge present (TYPE line)" \
    "$(echo "$BODY" | grep -q "# TYPE mlx_serve:memory_mb gauge" && echo 1 || echo 0)"
check "mlx_serve:generation_tokens_live gauge present (TYPE line)" \
    "$(echo "$BODY" | grep -q "# TYPE mlx_serve:generation_tokens_live gauge" && echo 1 || echo 0)"

# All counters must be 0 before any requests
check "request_success_total is 0 before any requests" \
    "$(echo "$BODY" | grep "^vllm:request_success_total " | grep -q " 0$" && echo 1 || echo 0)"
check "prompt_tokens_total is 0 before any requests" \
    "$(echo "$BODY" | grep "^vllm:prompt_tokens_total " | grep -q " 0$" && echo 1 || echo 0)"

# ════════════════════════════════════════════════════════════════════════════
# Test 3: After one chat request, counters are non-zero
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 3: after one chat completion ──"

CHAT=$(curl -s -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Reply with one word: OK"}],"max_tokens":5,"temperature":0}')

check "chat completion returned a response" \
    "$(echo "$CHAT" | grep -q '"choices"' && echo 1 || echo 0)"

BODY2=$(curl -s "$BASE/metrics")

# request_success_total must be 1
check "request_success_total == 1 after one request" \
    "$(echo "$BODY2" | grep "^vllm:request_success_total " | grep -q " 1$" && echo 1 || echo 0)"

# prompt_tokens_total must be > 0
PT=$(echo "$BODY2" | grep "^vllm:prompt_tokens_total " | awk '{print $2}')
check "prompt_tokens_total > 0 after one request" \
    "$([ -n "$PT" ] && [ "$PT" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"

# TTFT histogram count must be 1
check "vllm:time_to_first_token_seconds_count == 1" \
    "$(echo "$BODY2" | grep "^vllm:time_to_first_token_seconds_count " | grep -q " 1$" && echo 1 || echo 0)"

# e2e histogram count must be 1
check "vllm:e2e_request_latency_seconds_count == 1" \
    "$(echo "$BODY2" | grep "^vllm:e2e_request_latency_seconds_count " | grep -q " 1$" && echo 1 || echo 0)"

# +Inf bucket must be cumulative 1 (one request went through)
TTFT_INF=$(echo "$BODY2" | grep 'vllm:time_to_first_token_seconds_bucket{le="+Inf"}' | awk '{print $2}')
check "TTFT +Inf bucket == 1" \
    "$([ "$TTFT_INF" = "1" ] && echo 1 || echo 0)"

# request_cancelled_total must be 0 (no cancellations)
check "request_cancelled_total == 0" \
    "$(echo "$BODY2" | grep "^vllm:request_cancelled_total " | grep -q " 0$" && echo 1 || echo 0)"

# memory_mb must reflect loaded model footprint (phys_footprint, not resident_size)
# Any loaded model footprints >500 MB; the old resident_size bug read ~14 MB.
MEM=$(echo "$BODY2" | grep "^mlx_serve:memory_mb " | awk '{print $2}')
check "mlx_serve:memory_mb > 500 after model load (phys_footprint, not resident_size)" \
    "$([ -n "$MEM" ] && [ "$MEM" -gt 500 ] 2>/dev/null && echo 1 || echo 0)"

# generation_tokens_live (real-time tok/s source) = completed tokens + tokens
# generated so far by in-flight slots. The gauge sampler runs every 2s, so wait
# one cadence for it to reflect the just-completed request. With nothing
# decoding at scrape time it must equal generation_tokens_total and be > 0.
sleep 3
BODY3=$(curl -s "$BASE/metrics")
GEN=$(echo "$BODY3" | grep "^vllm:generation_tokens_total " | awk '{print $2}')
LIVE=$(echo "$BODY3" | grep "^mlx_serve:generation_tokens_live " | awk '{print $2}')
check "generation_tokens_live > 0 after one request (sampler ticked)" \
    "$([ -n "$LIVE" ] && [ "$LIVE" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
check "generation_tokens_live == generation_tokens_total at rest (no slots decoding)" \
    "$([ -n "$LIVE" ] && [ -n "$GEN" ] && [ "$LIVE" = "$GEN" ] && echo 1 || echo 0)"

# ── Summary ────────────────────────────────────────────────────────────────
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
