#!/usr/bin/env bash
# Tick-time A/B: two streams at ~64k and ~162k, MLX_SERVE_QSA_BATCHED_GATHER=0|1.
# Same-boot pair, 100s idle first, MLX_SERVE_ROUND_COST_PERSIST=0 on both arms.
# Counterbalance by running twice with GATHER_ORDER=1,0 then GATHER_ORDER=0,1.
# Usage: BENCH_MODEL=<qwen4_exp dir> ./tests/bench_batched_qsa_gather.sh [port]
set -euo pipefail
PORT=${1:-8098}
MODEL="${BENCH_MODEL:?set BENCH_MODEL to a qwen4_exp pack}"
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
ORDER="${GATHER_ORDER:-1,0}"
OUT="${BENCH_OUT:-$HOME/claude-tmp/qsa-batched-gather}"
mkdir -p "$OUT"
if [ ! -x "$BINARY" ]; then
    echo "FAIL $BINARY missing — build first"
    exit 1
fi
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
prompt_of() {
    python3 -c "print(('The history of computing. ' * int($1 * 4))[:int($1 * 16)])"
}
one() {
    local toks=$1
    python3 -c "
import json,sys,time
t0=time.time()
body=json.dumps({
    'model':'mlx-serve',
    'messages':[{'role':'user','content':sys.argv[1]}],
    'max_tokens': 64,
    'temperature': 0,
    'stream': False,
})
print(body)
" "$(prompt_of "$toks")" | curl -s -m 3600 -X POST -H 'Content-Type: application/json' -d @- "http://127.0.0.1:$PORT/v1/chat/completions" | python3 -c "
import sys,json
j=json.load(sys.stdin)
t=j.get('timings') or {}
n=t.get('predicted_n') or j.get('usage',{}).get('completion_tokens') or 0
ms=t.get('predicted_ms') or 0
pt=t.get('prompt_n') or j.get('usage',{}).get('prompt_tokens') or 0
tick = (ms/n) if n else 0
print(f\"prompt={pt} predicted={n} predicted_ms={ms:.1f} ms/tick={tick:.2f} tok/s={t.get('predicted_per_second',0):.1f}\")
"
}
run_arm() {
    local gather=$1
    local log="$OUT/gather_$gather.log"
    echo "== MLX_SERVE_QSA_BATCHED_GATHER=$gather =="
    MLX_SERVE_ROUND_COST_PERSIST=0 MLX_SERVE_QSA_BATCHED_GATHER=$gather \
        "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --max-concurrent 4 --prefix-cache-entries 0 --no-pld --log-level info \
        > "$log" 2>&1 &
    local pid=$!
    for i in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && grep -q "Model ready" "$log" && break
        sleep 1
    done
    echo "idle 100s"
    sleep 100
    local t0
    t0=$(python3 -c 'import time; print(time.time())')
    one 64000 > "$OUT/g${gather}_64k.out" &
    local p1=$!
    one 162000 > "$OUT/g${gather}_162k.out" &
    local p2=$!
    wait $p1 $p2
    echo "64k  $(cat "$OUT/g${gather}_64k.out")"
    echo "162k $(cat "$OUT/g${gather}_162k.out")"
    echo "wall $(python3 -c "import time; print(f'{time.time()-$t0:.1f}s')")"
    echo "batched engaged: $(grep -c '\[batched\].*engaged' "$log" || true)"
    echo "qsa-batched-gather: $(grep -c '\[qsa-batched-gather\] engaged' "$log" || true)"
    grep -E '\[batched\].*engaged|\[qsa-batched-gather\] engaged|\[qsa-arms\]|\[qsa-decode-gather\]|\[qsa-verify-gather\]' "$log" || true
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
    for i in $(seq 1 30); do
        lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || break
        sleep 1
    done
}
IFS=',' read -r -a arms <<< "$ORDER"
for g in "${arms[@]}"; do
    run_arm "$g"
done
