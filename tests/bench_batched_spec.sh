#!/usr/bin/env bash
# Step-4 measurement gate: per-stream decode tok/s (server `timings`) for
#   serial N=1, MTP N=1, plain batched N=2, and N=2 with MTP on one/both slots.
# Usage: BENCH_MODEL=<dir> [SPEC_FLAGS="--mtp"] ./tests/bench_batched_spec.sh [port] [max_tokens]
# The "mtp" arms enable whatever spec the server loaded (MTP head or DFlash drafter).
set -u
MODEL="${BENCH_MODEL:?set BENCH_MODEL}"
PORT=${1:-18820}; MAXTOK=${2:-256}; BASE="http://127.0.0.1:$PORT"
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
LOG=$(mktemp)
"$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld ${SPEC_FLAGS:---mtp} --max-concurrent 4 --prefix-cache-entries 0 > "$LOG" 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null; rm -f "$LOG"' EXIT
for i in $(seq 1 120); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
grep -E "Concurrency:|MTP|mtp" "$LOG" | head -3

PROMPTS=("Write a detailed essay about the history of quantum computing." "Explain how a compiler turns source code into machine code, in depth." "Describe the water cycle for a graduate seminar, with mechanisms.")
req() { # $1 prompt idx, $2 enable_mtp (true|false)
    python3 -c "
import json,sys
print(json.dumps({'model':'mlx-serve','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$MAXTOK,'temperature':0,'enable_mtp':sys.argv[2]=='true','enable_drafter':sys.argv[2]=='true','stream':False}))" "${PROMPTS[$1]}" "$2" |
    curl -s -m 600 -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" 2>/dev/null |
    python3 -c "import sys,json; d=sys.stdin.read(); j=json.loads(d) if d else {}; t=j.get('timings'); print(f\"{t['predicted_n']} tok {t['predicted_per_second']:.1f} tok/s\" if t else 'ERR '+d[:200])"
}
run() { # label, then pairs "idx:mtp"
    local label=$1; shift
    local pids=(); local i=0
    for spec in "$@"; do
        req "${spec%%:*}" "${spec##*:}" > "/tmp/bbs_$i.$$" & pids+=($!); i=$((i+1))
    done
    wait "${pids[@]}"
    echo "== $label"
    for j in $(seq 0 $((i-1))); do echo "   stream $j: $(cat /tmp/bbs_$j.$$)"; rm -f /tmp/bbs_$j.$$; done
    grep -c "slot serial" "$LOG" | sed 's/^/   serial-reason lines so far: /'
}
req 0 false >/dev/null  # warm
run "N=1 serial"        0:false
run "N=1 mtp"           0:true
run "N=2 plain batched" 0:false 1:false
run "N=2 mtp + plain"   0:true  1:false
run "N=2 mtp + mtp"     0:true  1:true
run "N=3 plain batched" 0:false 1:false 2:false
run "N=3 mtp x3"        0:true  1:true  2:true
grep -E "\[batched\]|\[spec-stats\]" "$LOG" | tail -6
