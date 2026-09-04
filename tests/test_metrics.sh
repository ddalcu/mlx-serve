#!/bin/bash
# Integration tests for the opt-in observability layer (--metrics):
#   * GET /metrics       — Prometheus text exposition (headless scraping)
#   * GET /metrics.json  — open JSON feed (drives the index-page panel)
#   * GET /              — index page hosts a live metrics panel when --metrics
#
# There is NO admin dashboard, NO auth, and NO admin mutations — the panel is
# open and read-only. (The old tests/test_admin_api.sh is retired.)
#
# Tests:
#  1. Without --metrics: /metrics + /metrics.json → 503; index page has no panel.
#  2. With    --metrics: /metrics → 200 Prometheus text; /metrics.json → 200 JSON;
#                        index page embeds the panel + polls /metrics.json.
#  3. After one chat request: counters/histograms increment; live-gauge holds
#                             (live > 0 after a request, live == total at rest).
#
# Usage: ./tests/test_metrics.sh [model_dir] [port]
#   Starts its own servers. Default model: Gemma 4 E4B 8-bit.

set -u

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
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
    echo "SKIP: model dir not found: $MODEL (pass as first arg)"
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

# ════════════════════════════════════════════════════════════════════════════
# Phase 0: index-panel rate math (pure, no server, no GPU)
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 0: panel rate math (src/html/metrics.js) ──"
NODE_BIN="$(command -v node || true)"
if [ -z "$NODE_BIN" ]; then
    echo "  SKIP: node not on PATH"
else
    if "$NODE_BIN" "$(dirname "$0")/metrics_panel_test.mjs" > /tmp/metrics_panel.out 2>&1; then
        sed 's/^/  /' /tmp/metrics_panel.out | grep -E "PASS|ALL PASS"
        check "panel rate math (no carry-forward; prefill 0 while decoding)" 1
    else
        sed 's/^/  /' /tmp/metrics_panel.out
        check "panel rate math (no carry-forward; prefill 0 while decoding)" 0
    fi
fi

# ════════════════════════════════════════════════════════════════════════════
# Phase 1: Without --metrics, /metrics* return 503 and the index page has no panel
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 1: without --metrics ──"

"$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT
wait_health

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics")
check "GET /metrics without --metrics → 503" "$([ "$STATUS" = "503" ] && echo 1 || echo 0)"

BODY=$(curl -s "$BASE/metrics")
check "503 body mentions 'not enabled'" "$(echo "$BODY" | grep -q "not enabled" && echo 1 || echo 0)"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics.json")
check "GET /metrics.json without --metrics → 503" "$([ "$STATUS" = "503" ] && echo 1 || echo 0)"

INDEX=$(curl -s "$BASE/")
check "index page renders (200-ish, console markup present)" \
    "$(echo "$INDEX" | grep -q 'data-tab="chat"' && echo 1 || echo 0)"
check "index page has NO metrics panel when --metrics off" \
    "$(echo "$INDEX" | grep -q 'id=m-status' && echo 0 || echo 1)"

kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true; sleep 1

# ════════════════════════════════════════════════════════════════════════════
# Phase 2: With --metrics, endpoints + index panel are present
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 2: with --metrics (idle — no requests yet) ──"

# --log-level info so the "Prometheus metrics: ENABLED" startup line is visible.
"$BINARY" --model "$MODEL" --serve --port "$PORT" --metrics --no-pld --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT
wait_health

check "startup log: 'Prometheus metrics: ENABLED'" \
    "$(grep -q "Prometheus metrics: ENABLED" "$LOG" && echo 1 || echo 0)"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics")
check "GET /metrics with --metrics → 200" "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"

CT=$(curl -s -D - -o /dev/null "$BASE/metrics" | grep -i "^content-type:" | tr -d '\r')
check "Content-Type is Prometheus text MIME" \
    "$(echo "$CT" | grep -q "text/plain" && echo "$CT" | grep -q "version=0.0.4" && echo 1 || echo 0)"

BODY=$(curl -s "$BASE/metrics")
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

check "request_success_total is 0 before any requests" \
    "$(echo "$BODY" | grep "^vllm:request_success_total " | grep -q " 0$" && echo 1 || echo 0)"
check "prompt_tokens_total is 0 before any requests" \
    "$(echo "$BODY" | grep "^vllm:prompt_tokens_total " | grep -q " 0$" && echo 1 || echo 0)"

# JSON feed shape
JSTATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics.json")
check "GET /metrics.json with --metrics → 200" "$([ "$JSTATUS" = "200" ] && echo 1 || echo 0)"
JCT=$(curl -s -D - -o /dev/null "$BASE/metrics.json" | grep -i "^content-type:" | tr -d '\r')
check "/metrics.json Content-Type is application/json" \
    "$(echo "$JCT" | grep -q "application/json" && echo 1 || echo 0)"
JBODY=$(curl -s "$BASE/metrics.json")
check "/metrics.json has 'counters' key"   "$(echo "$JBODY" | grep -q '"counters"' && echo 1 || echo 0)"
check "/metrics.json has 'gauges' key"     "$(echo "$JBODY" | grep -q '"gauges"' && echo 1 || echo 0)"
check "/metrics.json has 'histograms' key" "$(echo "$JBODY" | grep -q '"histograms"' && echo 1 || echo 0)"
check "/metrics.json has 'generation_tokens_live'" \
    "$(echo "$JBODY" | grep -q '"generation_tokens_live"' && echo 1 || echo 0)"
check "/metrics.json has 'bucket_counts'"  "$(echo "$JBODY" | grep -q '"bucket_counts"' && echo 1 || echo 0)"

# Index page hosts the live panel when --metrics is on
INDEX=$(curl -s "$BASE/")
check "index page HAS the metrics panel when --metrics on" \
    "$(echo "$INDEX" | grep -q 'id=m-status' && echo 1 || echo 0)"
check "index panel polls /metrics.json" \
    "$(echo "$INDEX" | grep -q "/metrics.json" && echo 1 || echo 0)"
check "index panel has decode + prefill tok/s tiles" \
    "$(echo "$INDEX" | grep -q 'm-decode-tps' && echo "$INDEX" | grep -q 'm-prefill-tps' && echo 1 || echo 0)"
check "index panel has decode + prefill sparklines" \
    "$(echo "$INDEX" | grep -q 'm-spark-decode' && echo "$INDEX" | grep -q 'm-spark-prefill' && echo 1 || echo 0)"

# ════════════════════════════════════════════════════════════════════════════
# Phase 3: After one chat request, counters are non-zero
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 3: after one chat completion ──"

CHAT=$(curl -s -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Reply with one word: OK"}],"max_tokens":5,"temperature":0}')

check "chat completion returned a response" \
    "$(echo "$CHAT" | grep -q '"choices"' && echo 1 || echo 0)"

BODY2=$(curl -s "$BASE/metrics")

check "request_success_total == 1 after one request" \
    "$(echo "$BODY2" | grep "^vllm:request_success_total " | grep -q " 1$" && echo 1 || echo 0)"

PT=$(echo "$BODY2" | grep "^vllm:prompt_tokens_total " | awk '{print $2}')
check "prompt_tokens_total > 0 after one request" \
    "$([ -n "$PT" ] && [ "$PT" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"

check "vllm:time_to_first_token_seconds_count == 1" \
    "$(echo "$BODY2" | grep "^vllm:time_to_first_token_seconds_count " | grep -q " 1$" && echo 1 || echo 0)"

check "vllm:e2e_request_latency_seconds_count == 1" \
    "$(echo "$BODY2" | grep "^vllm:e2e_request_latency_seconds_count " | grep -q " 1$" && echo 1 || echo 0)"

TTFT_INF=$(echo "$BODY2" | grep 'vllm:time_to_first_token_seconds_bucket{le="+Inf"}' | awk '{print $2}')
check "TTFT +Inf bucket == 1" \
    "$([ "$TTFT_INF" = "1" ] && echo 1 || echo 0)"

check "request_cancelled_total == 0" \
    "$(echo "$BODY2" | grep "^vllm:request_cancelled_total " | grep -q " 0$" && echo 1 || echo 0)"

# memory_mb must reflect the loaded model footprint (phys_footprint, not
# resident_size). Any loaded model footprints >500 MB.
MEM=$(echo "$BODY2" | grep "^mlx_serve:memory_mb " | awk '{print $2}')
check "mlx_serve:memory_mb > 500 (phys_footprint, not resident_size)" \
    "$([ -n "$MEM" ] && [ "$MEM" -gt 500 ] 2>/dev/null && echo 1 || echo 0)"

# generation_tokens_live (live tok/s source) = completed + in-flight. The gauge
# sampler ticks every 2s, so wait one cadence. With nothing decoding at scrape
# time it must equal generation_tokens_total AND be > 0.
sleep 3
BODY3=$(curl -s "$BASE/metrics")
GEN=$(echo "$BODY3" | grep "^vllm:generation_tokens_total " | awk '{print $2}')
LIVE=$(echo "$BODY3" | grep "^mlx_serve:generation_tokens_live " | awk '{print $2}')
check "generation_tokens_live > 0 after one request (sampler ticked)" \
    "$([ -n "$LIVE" ] && [ "$LIVE" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
check "generation_tokens_live == generation_tokens_total at rest (no slots decoding)" \
    "$([ -n "$LIVE" ] && [ -n "$GEN" ] && [ "$LIVE" = "$GEN" ] && echo 1 || echo 0)"

# ── Phase 4: prefill is visible WHILE it runs, not only when the request ends ──
#
# Regression: `prompt_tokens_total` and the prefill_time histogram only advance
# at request completion, and generated tokens only accrue during decode. So a
# multi-minute prefill pinned the GPU while the panel showed 0 tok/s decode and
# "—" prefill — the user could not tell a long prefill from a hung server.
# `mlx_serve:prefill_tokens_live` is the missing signal.
echo ""
echo "── Phase 4: live prefill gauge ──"

# At rest, no prefill is in flight.
sleep 3
IDLE_PRE=$(curl -s "$BASE/metrics" | grep "^mlx_serve:prefill_tokens_live " | awk '{print $2}')
check "prefill_tokens_live == 0 at rest" \
    "$([ "$IDLE_PRE" = "0" ] && echo 1 || echo 0)"

# Build a prompt big enough that prefill spans several chunks (chunk = 8192
# tokens), then poll the gauge WHILE the request is still in flight.
BIG=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * 2600).strip())")
REQ=$(python3 -c "
import json,sys
print(json.dumps({'model':'mlx-serve','stream':False,'max_tokens':1,'temperature':0,
                  'messages':[{'role':'user','content':sys.stdin.read()}]}))" <<< "$BIG")

curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" >/dev/null &
CURL_PID=$!

SAW_LIVE=0
SAW_PHASE=0
MAX_SEEN=0
for _ in $(seq 1 200); do
    kill -0 $CURL_PID 2>/dev/null || break     # request finished
    read -r V P <<< "$(curl -s -m 2 "$BASE/metrics.json" | python3 -c "
import json,sys
try:
    g = json.load(sys.stdin)['gauges']
    print(g.get('prefill_tokens_live', 0), g.get('requests_prefilling', 0))
except Exception: print(0, 0)" 2>/dev/null)"
    [ -n "$P" ] && [ "$P" -gt 0 ] 2>/dev/null && SAW_PHASE=1
    [ -n "$V" ] && [ "$V" -gt "$MAX_SEEN" ] 2>/dev/null && MAX_SEEN=$V
    [ -n "$V" ] && [ "$V" -gt 0 ] 2>/dev/null && SAW_LIVE=1 && break
    sleep 0.5
done
wait $CURL_PID 2>/dev/null

check "prefill_tokens_live > 0 DURING a long prefill (saw $MAX_SEEN tokens in flight)" "$SAW_LIVE"
check "requests_prefilling was 1 during the prefill (phase visible before the first chunk)" "$SAW_PHASE"

# ...and it returns to 0 once the prefill is done.
sleep 3
DONE_PRE=$(curl -s "$BASE/metrics" | grep "^mlx_serve:prefill_tokens_live " | awk '{print $2}')
DONE_PHASE=$(curl -s "$BASE/metrics" | grep "^mlx_serve:requests_prefilling " | awk '{print $2}')
check "prefill_tokens_live back to 0 after the request completes" \
    "$([ "$DONE_PRE" = "0" ] && echo 1 || echo 0)"
check "requests_prefilling back to 0 after the request completes" \
    "$([ "$DONE_PHASE" = "0" ] && echo 1 || echo 0)"

# ── Phase 5: prefill throughput must exclude prefix-cache restores ──
#
# `prompt_tokens_total` bills every prompt token; `prefill_time_seconds` only
# ticks for tokens actually forwarded. Dividing the first by the second inflated
# the panel's prefill tok/s by prompt/(prompt-cached) — 10.6x on a warm
# multi-turn 35B MoE session (9.8K tok/s reported, ~220 real).
echo ""
echo "── Phase 5: prefill tok/s excludes cached tokens ──"

read_counter() { curl -s "$BASE/metrics" | grep "^$1 " | awk '{print $2}'; }

# Same prompt twice: the second request must hit the hot prefix cache.
WARM='{"model":"mlx-serve","max_tokens":1,"temperature":0,"messages":[{"role":"user","content":"Count slowly and describe each number in one clause: one two three four five six seven eight nine ten eleven twelve."}]}'
curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$WARM" >/dev/null
P1=$(read_counter "vllm:prompt_tokens_total")
F1=$(read_counter "mlx_serve:prefill_tokens_total")

curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$WARM" >/dev/null
P2=$(read_counter "vllm:prompt_tokens_total")
F2=$(read_counter "mlx_serve:prefill_tokens_total")
C2=$(read_counter "mlx_serve:prefix_cache_tokens_total")

DP=$((P2 - P1))   # billed prompt tokens of the warm request
DF=$((F2 - F1))   # tokens it actually forwarded

check "warm request still bills its full prompt (prompt_tokens_total +$DP)" \
    "$([ "$DP" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
check "warm request forwards FEWER tokens than it bills ($DF < $DP)" \
    "$([ "$DF" -lt "$DP" ] 2>/dev/null && echo 1 || echo 0)"
check "prefix_cache_tokens_total > 0 after a cache hit" \
    "$([ -n "$C2" ] && [ "$C2" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
# The invariant that makes prefill tok/s trustworthy.
check "forwarded + restored == billed ($F2 + $C2 == $P2)" \
    "$([ $((F2 + C2)) -eq "$P2" ] 2>/dev/null && echo 1 || echo 0)"

# The panel divides by this counter; it must never exceed the billed total.
check "prefill_tokens_total <= prompt_tokens_total" \
    "$([ "$F2" -le "$P2" ] 2>/dev/null && echo 1 || echo 0)"

# ── Phase 6: the prefill target is the POST-trim figure ──
#
# `prefill_prompt_tokens` is published at the top of the MLX prefill (the
# UNTRIMMED prompt); `prefill_target_tokens` only after the hot prefix cache
# has trimmed the reused head off it. Their difference is what the cache
# restored for the request in flight.
#
# This phase is the guard the unit test cannot be. Dropping the target store,
# or moving it back above the trim, still renders a well-formed gauge and still
# passes `zig build test` — what it breaks is only visible live: target ==
# prompt on a warm request, the reuse invisible, and the progress bar jumping
# to ~97% on the first chunk.
echo ""
echo "── Phase 6: prefill target is post-trim (live prefix-cache reuse) ──"

# Sample the pair WHILE a request prefills, keeping only a POST-TRIM sample.
#
# The gauge sampler ticks every 2 s and the target is published later than the
# prompt (the cache restore runs between them), so an early tick legitimately
# reads "prompt=N target=0". Latching the first sample with the largest prompt
# would keep exactly that one and report target=0 for every request — so take
# the largest prompt among samples that HAVE a target. `final_len` is at least
# 1 even on a full match (the cache re-forwards the last token), so target > 0
# is a sound "we are past the trim" signal. Echoes "prompt target", or "0 0"
# when the window was missed entirely.
sample_pair() {
    local body="$1" bp=0 bt=0 pp tt
    curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
         -d "$body" >/dev/null &
    local pid=$!
    for _ in $(seq 1 700); do
        kill -0 $pid 2>/dev/null || break
        read -r pp tt <<< "$(curl -s -m 2 "$BASE/metrics.json" | python3 -c "
import json,sys
try:
    g = json.load(sys.stdin)['gauges']
    print(g.get('prefill_prompt_tokens', 0), g.get('prefill_target_tokens', 0))
except Exception: print(0, 0)" 2>/dev/null)"
        if [ "${tt:-0}" -gt 0 ] 2>/dev/null && [ "${pp:-0}" -ge "$bp" ] 2>/dev/null; then
            bp=$pp; bt=$tt
        fi
        sleep 0.5
    done
    wait $pid 2>/dev/null
    echo "$bp $bt"
}

mk_body() { python3 -c "
import json,sys
print(json.dumps({'model':'mlx-serve','stream':False,'max_tokens':1,'temperature':0,
                  'messages':[{'role':'user','content':sys.stdin.read()}]}))" ; }

# One shared head, a UNIQUE tail per request. A warm request then reuses
# exactly the head and forwards exactly its own tail — which keeps the warm
# prefill long enough (thousands of tokens) to span a 2 s sampler tick. Re-
# sending an identical prompt would instead trim to a single token and finish
# between two ticks.
P6_HEAD=$(python3 -c "print(('A prefix cache restores the head of a repeated prompt. ' * 1500).strip())")
p6_tail() { python3 -c "
import sys
print('Variant ' + sys.argv[1] + ' begins here. ' +
      ('This suffix is new and must actually be computed. ' * 600).strip())" "$1"; }

COLD=$(printf '%s' "$P6_HEAD" | mk_body)
read -r C_PROMPT C_TARGET <<< "$(sample_pair "$COLD")"
check "cold prefill: both gauges published mid-prefill (prompt=$C_PROMPT target=$C_TARGET)" \
    "$([ "$C_PROMPT" -gt 0 ] 2>/dev/null && [ "$C_TARGET" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
# Not `==`: earlier phases left entries in the cache that share this prompt's
# chat-template header, and the RAM tier has no minimum match length, so a
# handful of header tokens can legitimately be trimmed. What must hold is that
# nothing SUBSTANTIAL was reused.
check "cold prefill: nothing substantial reused, target >= 95% of prompt" \
    "$([ "$C_TARGET" -gt 0 ] 2>/dev/null && [ "$C_TARGET" -ge $((C_PROMPT * 95 / 100)) ] 2>/dev/null && echo 1 || echo 0)"

# Same head + a fresh tail: the cache restores the head, so the target must be
# MUCH smaller than the prompt. This is the assertion that dies if the target
# is published before the trim. Retry with a new tail if the sampler missed the
# post-trim window — the head stays hot, so every attempt is equivalent.
W_PROMPT=0; W_TARGET=0
for V in 1 2 3; do
    WARM6=$(p6_tail "$V" | { printf '%s ' "$P6_HEAD"; cat; } | mk_body)
    read -r W_PROMPT W_TARGET <<< "$(sample_pair "$WARM6")"
    [ "${W_TARGET:-0}" -gt 0 ] 2>/dev/null && break
done
W_REUSED=$((W_PROMPT - W_TARGET))
check "warm prefill: post-trim sample caught (prompt=$W_PROMPT target=$W_TARGET)" \
    "$([ "$W_TARGET" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
check "warm prefill: prompt grew (=$W_PROMPT > cold $C_PROMPT)" \
    "$([ "$W_PROMPT" -gt "$C_PROMPT" ] 2>/dev/null && echo 1 || echo 0)"
check "warm prefill: target is POST-trim, well below prompt ($W_TARGET < half of $W_PROMPT)" \
    "$([ "$W_TARGET" -lt $((W_PROMPT / 2)) ] 2>/dev/null && echo 1 || echo 0)"
check "warm prefill: reuse is visible and matches the cold prompt (reused=$W_REUSED ~ $C_PROMPT)" \
    "$([ "$W_REUSED" -gt $((C_PROMPT * 8 / 10)) ] 2>/dev/null && echo 1 || echo 0)"

# Both return to rest, so "prefilling" keeps meaning exactly that.
sleep 3
IDLE_PAIR=$(curl -s "$BASE/metrics.json" | python3 -c "
import json,sys
g = json.load(sys.stdin)['gauges']
print(g.get('prefill_prompt_tokens', -1), g.get('prefill_target_tokens', -1))")
check "both gauges back to 0 at rest ($IDLE_PAIR)" \
    "$([ "$IDLE_PAIR" = "0 0" ] && echo 1 || echo 0)"

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
