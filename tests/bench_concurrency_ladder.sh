#!/usr/bin/env bash
# Concurrency ladder: client-observed per-stream and aggregate decode tok/s at N
# concurrent users, MTP on and off. One CSV row per (binary, model, mode, N).
# Usage: BENCH_MODEL=<dir> [BENCH_LABEL=x] [MLX_SERVE_BINARY=..] ./tests/bench_concurrency_ladder.sh <port> <csv> ["1 2 3 4 6 8"] [max_tokens]
set -u
MODEL="${BENCH_MODEL:?}"; PORT=$1; CSV=$2; NS="${3:-1 2 3 4 6 8}"; MAXTOK=${4:-200}
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"; LABEL="${BENCH_LABEL:-$(basename $(dirname $(dirname $BINARY)))}"
LOG=$(mktemp)
MLX_SERVE_ROUND_COST_PERSIST=0 "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --mtp --max-concurrent 16 --prefix-cache-entries 0 > "$LOG" 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null; rm -f "$LOG"' EXIT
for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
P=("Write a short essay about the history of quantum computing." "Explain how a compiler turns source code into machine code." "Describe the water cycle for a graduate seminar." "Summarize the causes of the French revolution." "Write a poem about the sea in the style of Whitman." "Explain TCP congestion control to a junior engineer." "Describe how vaccines train the immune system." "Outline a beginner's plan to learn the piano.")
req() { # prompt mtp -> "tokens wall_s"
    local t0=$(python3 -c 'import time; print(time.time())')
    local n=$(python3 -c "import json,sys; print(json.dumps({'model':'mlx-serve','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$MAXTOK,'temperature':0,'enable_mtp':sys.argv[2]=='true','enable_drafter':sys.argv[2]=='true'}))" "$1" "$2" |
        curl -s -m 1800 -X POST -H 'Content-Type: application/json' -d @- "http://127.0.0.1:$PORT/v1/chat/completions" | python3 -c "import sys,json; j=json.load(sys.stdin); print(j['usage']['completion_tokens'])")
    local t1=$(python3 -c 'import time; print(time.time())')
    echo "$n $(python3 -c "print($t1-$t0)")"
}
[ -s "$CSV" ] || echo "binary,model,mode,n,per_stream_tps,aggregate_tps,min_stream_tps" > "$CSV"
req "${P[0]}" true >/dev/null
for mode in true false; do
  for N in $NS; do
    T0=$(python3 -c 'import time; print(time.time())')
    pids=(); for i in $(seq 0 $((N-1))); do req "${P[$((i % 8))]} (variant $i)" $mode > "/tmp/bcl_$$_$i" & pids+=($!); done; wait "${pids[@]}"
    T1=$(python3 -c 'import time; print(time.time())')
    row=$(cat /tmp/bcl_$$_* | python3 -c "
import sys
rows=[l.split() for l in sys.stdin.read().strip().splitlines()]
tps=[int(n)/float(w) for n,w in rows]; tot=sum(int(n) for n,_ in rows)
print(f'{sum(tps)/len(tps):.1f},{tot/($T1-$T0):.1f},{min(tps):.1f}')"); rm -f /tmp/bcl_$$_*
    echo "$LABEL,$(basename "$MODEL"),$([ $mode = true ] && echo mtp || echo plain),$N,$row" | tee -a "$CSV"
  done
done
