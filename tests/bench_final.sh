#!/bin/bash
# bench_final.sh — focused, robust mlx-serve vs LM Studio comparison.
#
# For each (model, engine) pair: send the SAME long prompt 5 times. Engine
# stays loaded the whole time; run 1 is cold, runs 2-5 are warm. Median of
# warm runs is the warm number; run 1 is the cold number. No process restarts
# in the middle (which was breaking the prior harness when LMS choked on
# repeated JIT loads).
#
# Output:
#   docs/perf-csvs/bench_final-YYYYMMDD-HHMM.csv  (label|kind|run|prompt_n|cached_n|prompt_ms|wall_ms|hw)
#   docs/perf-vs-lmstudio-final-YYYYMMDD-HHMM.png
#
# Each (label, kind) line is run sequentially with a clean cache so prior
# state doesn't leak. mlx-serve is restarted between (model) but not between
# (run). LMS is JIT-loaded once per (model) via warmup ping.
set -uo pipefail

OUT_CSV="${1:-docs/perf-csvs/bench_final-$(date +%Y%m%d-%H%M).csv}"
RUNS=5

BIN="${BINARY:-./zig-out/bin/mlx-serve}"
PORT_SERVE=19099
PORT_LMS=1234
CTX=8192

mkdir -p "$(dirname "$OUT_CSV")"
[[ -x "$BIN" ]] || { echo "build mlx-serve first" >&2; exit 1; }
command -v jq >/dev/null

HW="$(sysctl -n machdep.cpu.brand_string | tr ' ' '-')"
RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
HW_TAG="${HW}-${RAM_GB}gb"

echo "label|kind|run|prompt_n|cached_n|prompt_ms|wall_ms|hw" > "$OUT_CSV"

PROMPT="$(python3 - <<'PY'
para = ("The kingdom of Avalon was beset by trials. Each season brought new "
        "challenges to its people, but the king remained steadfast. ")
print("Continue the story in 2 short sentences.\n\nBackground:\n" + para * 50)
PY
)"

SERVER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 mlx-serve 2>/dev/null
    lms unload --all >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

start_mlx() {
    local path="$1"
    cleanup
    sleep 2
    "$BIN" --model "$path" --serve --port "$PORT_SERVE" --ctx-size "$CTX" \
        --prefix-cache-entries 4 --log-level warn > /tmp/bench_final.log 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -sf --max-time 2 "http://127.0.0.1:$PORT_SERVE/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "mlx-serve died:" >&2; tail -10 /tmp/bench_final.log >&2; return 1; }
        sleep 0.5
    done
    return 1
}

start_lms() {
    local model_id="$1"
    cleanup
    sleep 3
    # JIT-load via warmup. Long timeout because cold-load + first-token can take a minute.
    local body=$(jq -nc --arg m "$model_id" '{model:$m,messages:[{role:"user",content:"hi"}],max_tokens:1,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    curl -sf -m 600 -X POST "http://127.0.0.1:$PORT_LMS/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" >/dev/null
}

run_one() {
    local kind="$1" model="$2" port="$3"
    local body=$(jq -nc --arg m "$model" --arg p "$PROMPT" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:8,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    local t0 t1 resp
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    resp=$(curl -sf -m 120 -X POST "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    if [[ -z "$resp" ]]; then echo "ERR|0|0|0|0"; return; fi
    python3 - "$resp" "$t1" "$t0" <<'PY'
import json, sys
try:
    r = json.loads(sys.argv[1])
except Exception:
    print("ERR|0|0|0|0"); sys.exit(0)
wall = int(sys.argv[2]) - int(sys.argv[3])
u = r.get('usage', {}) or {}
t = r.get('timings', {}) or {}
print(f"{u.get('prompt_tokens',0)}|{t.get('cached_n',0)}|{t.get('prompt_ms',0)}|{wall}")
PY
}

emit() {
    local label="$1" kind="$2" run="$3" pn="$4" cn="$5" pms="$6" wall="$7"
    echo "${label}|${kind}|${run}|${pn}|${cn}|${pms}|${wall}|${HW_TAG}" | tee -a "$OUT_CSV"
}

bench_pair() {
    local label="$1" kind="$2" model_path_or_id="$3"
    echo "==> $label / $kind" >&2
    local port
    case "$kind" in
        mlx-serve)
            start_mlx "$model_path_or_id" || { echo "  start failed" >&2; return; }
            port="$PORT_SERVE"
            ;;
        lmstudio)
            start_lms "$model_path_or_id" || { echo "  load failed" >&2; return; }
            port="$PORT_LMS"
            ;;
    esac

    for run in $(seq 1 "$RUNS"); do
        IFS='|' read -r pn cn pms wall < <(run_one "$kind" "$model_path_or_id" "$port")
        if [[ "$pn" == "ERR" ]]; then
            echo "  run $run: ERR (retrying once)" >&2
            sleep 2
            IFS='|' read -r pn cn pms wall < <(run_one "$kind" "$model_path_or_id" "$port")
        fi
        echo "  run $run: prompt_n=$pn cached_n=$cn prompt_ms=$pms wall=$wall ms" >&2
        emit "$label" "$kind" "$run" "$pn" "$cn" "$pms" "$wall"
    done
}

bench_pair "qwen35-4b-mlx"  "mlx-serve" "/Volumes/Sandisk_1TB/Models/mlx-community/Qwen3.5-4B-MLX-4bit"
bench_pair "qwen35-4b-mlx"  "lmstudio"  "qwen3.5-4b-mlx"
bench_pair "gemma4-e4b"     "mlx-serve" "/Users/david/.mlx-serve/models/gemma-4-e4b-it-4bit"
bench_pair "gemma4-e4b"     "lmstudio"  "gemma-4-e4b-it-mlx"
bench_pair "qwen35-4b-gguf" "mlx-serve" "/Volumes/Sandisk_1TB/Models/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-IQ4_NL.gguf"
bench_pair "qwen35-4b-gguf" "lmstudio"  "qwen3.5-4b"

cleanup
echo
echo "CSV: $OUT_CSV"
