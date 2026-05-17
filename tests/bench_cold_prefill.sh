#!/bin/bash
# bench_cold_prefill.sh — measure cold first-request latency with and without
# --warmup-eager. Validates plan 04 Phase 1 (eager weight residency at load).
#
# Methodology: kill any running mlx-serve, restart with the chosen warmup
# setting, fire ONE request and record the wall time. The model's weights
# are not yet GPU-resident at start; the first request's `mlx_eval` would
# normally pay 600-900 ms of page-faulting cost on Gemma 4 E4B 4-bit. With
# --warmup-eager (default) the boot path pre-faults them so the first
# request only pays normal prefill cost.
#
# Acceptance: warm cold-first-request must be ≥ 2× faster than no-warm.
# Bench is repeated N times (default 3) per setting; take the median.
#
# Usage:
#   ./tests/bench_cold_prefill.sh [model_dir] [port] [runs]
#
# Defaults: model=~/.mlx-serve/models/gemma-4-e4b-it-4bit, port=19030, runs=3.

set -uo pipefail

MODEL_DIR="${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}"
PORT="${2:-19030}"
RUNS="${3:-3}"

BINARY="${BINARY:-./zig-out/bin/mlx-serve}"

[[ -x "$BINARY" ]] || { echo "Build first: zig build -Doptimize=ReleaseFast" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "Model not found: $MODEL_DIR" >&2; exit 1; }

trap 'pkill -9 -x mlx-serve 2>/dev/null; true' EXIT

# Median of integers via python3 (kept consistent with bench_vs_lmstudio.sh style).
median() {
    python3 -c "
import sys
xs = sorted([int(x) for x in sys.argv[1].split(',') if x])
print(xs[len(xs)//2] if xs else 0)
" "$1"
}

run_one_cold_request() {
    local warm_flag="$1"
    pkill -9 -x mlx-serve 2>/dev/null
    sleep 1

    "$BINARY" --model "$MODEL_DIR" --serve --port "$PORT" --ctx-size 4096 \
        --log-level warn $warm_flag > /tmp/bench_cold_prefill_engine.log 2>&1 &
    local pid=$!

    # Wait for /health to come up (model loaded; warmup runs before this point if enabled).
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 0.5
        kill -0 "$pid" 2>/dev/null || { echo "ERR: server died (see /tmp/bench_cold_prefill_engine.log)" >&2; return 1; }
    done

    local body
    body=$(jq -nc '{model:"x",messages:[{role:"user",content:"Hello"}],max_tokens:4,temperature:0.0,stream:false,enable_thinking:false}')
    local t0 t1
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    curl -sf -m 30 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body" >/dev/null
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')

    pkill -9 -x mlx-serve 2>/dev/null
    sleep 1
    echo "$((t1 - t0))"
}

echo "=== Cold first-request bench (model=$(basename "$MODEL_DIR"), runs=$RUNS) ==="
echo

warm_csv=""
nowarm_csv=""

for run in $(seq 1 "$RUNS"); do
    echo "--- run $run/$RUNS ---"
    w=$(run_one_cold_request "--warmup-eager")
    echo "  --warmup-eager:    ${w} ms"
    warm_csv="${warm_csv}${w},"

    nw=$(run_one_cold_request "--no-warmup-eager")
    echo "  --no-warmup-eager: ${nw} ms"
    nowarm_csv="${nowarm_csv}${nw},"
done

WARM_MED=$(median "${warm_csv%,}")
NOWARM_MED=$(median "${nowarm_csv%,}")
echo
echo "=== medians ==="
echo "  warm:    ${WARM_MED} ms"
echo "  no-warm: ${NOWARM_MED} ms"

if [[ "$WARM_MED" -le 0 || "$NOWARM_MED" -le 0 ]]; then
    echo "FAIL: invalid timing measurements" >&2
    exit 1
fi

# Acceptance: warmup should at least halve cold-first-request wall time.
RATIO_X100=$((NOWARM_MED * 100 / WARM_MED))
echo "  ratio:   ${RATIO_X100}/100  (warmup is $((RATIO_X100 / 100)).$(printf '%02d' $((RATIO_X100 % 100)))× faster)"
if [[ "$RATIO_X100" -lt 200 ]]; then
    echo "FAIL: warmup gave less than 2× speedup" >&2
    exit 1
fi
echo "PASS: warmup ≥ 2× faster cold first request"
