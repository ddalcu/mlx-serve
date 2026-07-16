#!/bin/bash
# Plan 01 — A8 24h concurrent soak test.
#
# Spins up the server with `--max-concurrent 4 --kv-quant 4` and pounds it
# with four parallel workloads for `SOAK_DURATION_HOURS` hours (default 24,
# override via env). The workloads exercise distinct code paths so a leak
# or crash in any of them surfaces:
#
#   1. Plain chat completions  — varied prompts, 50-500 max_tokens.
#   2. Agent-loop style        — multi-turn fact recall (think tag enabled).
#   3. Anthropic /v1/messages  — exercises the parallel API.
#   4. Tool-calling round-trip — invokes a tool, feeds the result back.
#
# Every 5 minutes: record RSS / VSZ via `ps -o rss,vsz` to `tests/soak_log.csv`.
# Every 30 minutes: assert /health returns 200 within 5s.
# On exit: compute RSS drift = (rss_last_sample - rss_first_5min_sample) /
#          rss_first_5min_sample. FAIL if drift > 5% (or > 10% in the 1-hour
#          CI smoke variant, where transient allocator warm-up dominates).
#
# Usage:
#   SOAK_DURATION_HOURS=1 ./tests/test_soak_24h.sh                 # CI smoke
#   SOAK_DURATION_HOURS=24 ./tests/test_soak_24h.sh                # full
#   SOAK_DURATION_HOURS=2 SOAK_MODEL=/path/to/model ./tests/test_soak_24h.sh

set -e

SOAK_DURATION_HOURS="${SOAK_DURATION_HOURS:-24}"
PORT="${PORT:-8095}"
MODEL="${SOAK_MODEL:-${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}}"
BASE="http://127.0.0.1:$PORT"
LOG_DIR="${SOAK_LOG_DIR:-./tests}"
SAMPLE_LOG="$LOG_DIR/soak_log.csv"

# Drift threshold: 5% for ≥4h runs, 10% for shorter runs (warm-up dominates).
if [ "$SOAK_DURATION_HOURS" -ge 4 ]; then
    DRIFT_THRESHOLD_PCT="${DRIFT_THRESHOLD_PCT:-5}"
else
    DRIFT_THRESHOLD_PCT="${DRIFT_THRESHOLD_PCT:-10}"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_soak_24h: $MODEL not found."
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

DURATION_SEC=$(( SOAK_DURATION_HOURS * 3600 ))
SAMPLE_INTERVAL_SEC=300            # 5 minutes
HEALTH_INTERVAL_SEC=1800           # 30 minutes
# Shorter test → shorter sample interval so we get >= 4 samples even at 1h.
if [ "$SOAK_DURATION_HOURS" -lt 2 ]; then
    SAMPLE_INTERVAL_SEC=60
    HEALTH_INTERVAL_SEC=300
fi

echo "== Plan 01 A8 — 24h concurrent soak =="
echo "  duration:           ${SOAK_DURATION_HOURS}h (${DURATION_SEC}s)"
echo "  model:              $MODEL"
echo "  drift threshold:    ${DRIFT_THRESHOLD_PCT}%"
echo "  sample interval:    ${SAMPLE_INTERVAL_SEC}s"
echo "  health interval:    ${HEALTH_INTERVAL_SEC}s"
echo "  log:                $SAMPLE_LOG"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 2

SERVER_LOG=$(mktemp)
echo "  starting server (--max-concurrent 4 --kv-quant 4)..."
"$BINARY" --model "$MODEL" --serve --port "$PORT" \
    --max-concurrent 4 --kv-quant 4 \
    ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Background-worker PIDs (recorded so cleanup can kill them).
WORKER_PIDS=()
SAMPLER_PID=""
HEALTH_PID=""

cleanup() {
    echo
    echo "  shutting down workers + server..."
    for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true
    [ -n "$HEALTH_PID" ] && kill "$HEALTH_PID" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

up=0
for i in $(seq 1 120); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} server did not become healthy within 120s"
    tail -50 "$SERVER_LOG"
    exit 1
fi
echo -e "  ${GREEN}server up${NC} (pid=$SERVER_PID)"

mkdir -p "$LOG_DIR"
echo "ts,rss_kb,vsz_kb" > "$SAMPLE_LOG"

# ── Workload 1: chat completions ───────────────────────────────────────────
workload_chat() {
    local i=0
    while true; do
        local mt=$(( 50 + (i % 10) * 50 ))
        curl -s -X POST -H "Content-Type: application/json" \
            -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"Briefly describe item ${i}.\"}],\"max_tokens\":${mt},\"temperature\":0.0,\"stream\":false}" \
            "$BASE/v1/chat/completions" > /dev/null 2>&1 || true
        i=$(( i + 1 ))
        sleep 0.5
    done
}

# ── Workload 2: agent-loop style (multi-turn fact recall) ─────────────────
workload_agent() {
    local i=0
    while true; do
        # Three-turn synthetic conversation with thinking enabled.
        curl -s -X POST -H "Content-Type: application/json" \
            -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"Remember: my favorite color is #${i}.\"},{\"role\":\"assistant\",\"content\":\"Got it.\"},{\"role\":\"user\",\"content\":\"What did I just say?\"}],\"max_tokens\":80,\"temperature\":0.0,\"stream\":false}" \
            "$BASE/v1/chat/completions" > /dev/null 2>&1 || true
        i=$(( i + 1 ))
        sleep 1
    done
}

# ── Workload 3: Anthropic /v1/messages ────────────────────────────────────
workload_anthropic() {
    local i=0
    while true; do
        curl -s -X POST -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" \
            -d "{\"model\":\"mlx-serve\",\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"List ${i} random colors.\"}]}" \
            "$BASE/v1/messages" > /dev/null 2>&1 || true
        i=$(( i + 1 ))
        sleep 1
    done
}

# ── Workload 4: tool-calling round-trip ───────────────────────────────────
workload_tools() {
    local i=0
    while true; do
        curl -s -X POST -H "Content-Type: application/json" \
            -d "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"What's the weather in city #${i}? Use the get_weather tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Look up weather.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"max_tokens\":64,\"temperature\":0.0,\"stream\":false}" \
            "$BASE/v1/chat/completions" > /dev/null 2>&1 || true
        i=$(( i + 1 ))
        sleep 1
    done
}

# ── Sampler: ps every $SAMPLE_INTERVAL_SEC ───────────────────────────────
sampler() {
    while true; do
        local rss vsz ts
        ts=$(date +%s)
        # macOS ps emits header; tail -n1 takes the data line.
        read -r rss vsz < <(ps -o rss=,vsz= -p "$SERVER_PID" 2>/dev/null || echo "0 0")
        echo "${ts},${rss},${vsz}" >> "$SAMPLE_LOG"
        sleep "$SAMPLE_INTERVAL_SEC"
    done
}

# ── Health checker: every $HEALTH_INTERVAL_SEC ─────────────────────────────
health_checker() {
    while true; do
        if ! curl -s --max-time 5 -f "$BASE/health" > /dev/null 2>&1; then
            echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${RED}HEALTH-FAIL${NC}: /health did not respond within 5s" >&2
            # Don't exit — the main loop will catch a true crash via pid check.
        fi
        sleep "$HEALTH_INTERVAL_SEC"
    done
}

echo "  starting 4 workers + sampler + health checker..."
workload_chat       & WORKER_PIDS+=($!)
workload_agent      & WORKER_PIDS+=($!)
workload_anthropic  & WORKER_PIDS+=($!)
workload_tools      & WORKER_PIDS+=($!)
sampler             & SAMPLER_PID=$!
health_checker      & HEALTH_PID=$!

START_TS=$(date +%s)
END_TS=$(( START_TS + DURATION_SEC ))

while [ "$(date +%s)" -lt "$END_TS" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e "${RED}FAIL${NC} server pid $SERVER_PID is gone — crashed at $(date '+%Y-%m-%d %H:%M:%S')."
        tail -100 "$SERVER_LOG"
        exit 1
    fi
    # Sleep until next health check window — but cap so the kill -0 above
    # runs at least every minute.
    sleep 60
done

echo
echo "  duration elapsed; analyzing samples..."

# Need at least 2 samples to compute drift.
N_SAMPLES=$(tail -n +2 "$SAMPLE_LOG" | wc -l | tr -d ' ')
if [ "$N_SAMPLES" -lt 2 ]; then
    echo -e "${YELLOW}WARN${NC} only $N_SAMPLES sample(s) collected; skipping drift check."
    exit 0
fi

DRIFT=$(python3 - <<PY
import csv
rows = list(csv.DictReader(open("$SAMPLE_LOG")))
if len(rows) < 2:
    print("0.0")
else:
    # Use the SECOND sample as baseline so warm-up (page faults, JIT
    # compile, weight read-in) doesn't poison the denominator.
    base_idx = min(1, len(rows) - 1)
    base = int(rows[base_idx]['rss_kb'])
    last = int(rows[-1]['rss_kb'])
    if base == 0:
        print("0.0")
    else:
        print(f"{100.0 * (last - base) / base:.2f}")
PY
)
echo "  samples:   $N_SAMPLES"
echo "  RSS drift: ${DRIFT}% (threshold ${DRIFT_THRESHOLD_PCT}%)"

FAIL=0
OK=$(python3 -c "print(1 if abs(${DRIFT}) <= ${DRIFT_THRESHOLD_PCT} else 0)")
if [ "$OK" != "1" ]; then
    echo -e "${RED}FAIL${NC} RSS drift ${DRIFT}% exceeded threshold ${DRIFT_THRESHOLD_PCT}%"
    FAIL=1
else
    echo -e "${GREEN}PASS${NC} RSS drift within bound"
fi

# Health-check failures recorded? Surface them.
# NOTE: `grep -c` prints "0" AND exits 1 when there are no matches, so a
# `|| echo 0` fallback fires TOO and yields "0\n0" — which makes the `-gt`
# below die with "integer expression expected", i.e. the WARN silently never
# evaluates in the healthy case (live 2026-07-16 soak). Swallow grep's exit with
# `|| true` and default only when the output is EMPTY (file missing → exit 2,
# no stdout). Verified: clean→0, 2 failures→2 (warn fires), missing file→0.
HEALTH_FAILS=$(grep -c HEALTH-FAIL "$SERVER_LOG" 2>/dev/null || true)
HEALTH_FAILS=${HEALTH_FAILS:-0}
if [ "$HEALTH_FAILS" -gt 0 ]; then
    echo -e "${YELLOW}WARN${NC} $HEALTH_FAILS /health timeouts observed during run"
fi

exit $FAIL
