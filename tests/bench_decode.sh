#!/bin/bash
# bench_decode.sh — pure decode tok/s comparison. Short prompt, max_tokens=128,
# 3 runs per (model, engine), median reported. Companion to bench_final.sh
# which measures TTFT.
set -uo pipefail

OUT_CSV="${1:-docs/perf-csvs/bench_decode-$(date +%Y%m%d-%H%M).csv}"
RUNS=3
MAX_TOKENS=128

BIN="${BINARY:-./zig-out/bin/mlx-serve}"
PORT_SERVE=19099
PORT_LMS=1234

mkdir -p "$(dirname "$OUT_CSV")"
[[ -x "$BIN" ]] || { echo "build first" >&2; exit 1; }
command -v jq >/dev/null

HW="$(sysctl -n machdep.cpu.brand_string | tr ' ' '-')"
RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
HW_TAG="${HW}-${RAM_GB}gb"

echo "label|kind|run|completion_tokens|decode_tok_s|wall_ms|hw" > "$OUT_CSV"

PROMPT="Write a detailed essay about quantum computing."

SERVER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 mlx-serve 2>/dev/null
    lms unload --all >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

start_mlx() {
    cleanup; sleep 2
    "$BIN" --model "$1" --serve --port "$PORT_SERVE" --ctx-size 4096 --log-level warn > /tmp/bench_decode.log 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -sf --max-time 2 "http://127.0.0.1:$PORT_SERVE/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || return 1
        sleep 0.5
    done
    return 1
}

start_lms() {
    cleanup; sleep 3
    local body=$(jq -nc --arg m "$1" '{model:$m,messages:[{role:"user",content:"hi"}],max_tokens:1,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    curl -sf -m 600 -X POST "http://127.0.0.1:$PORT_LMS/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" >/dev/null
}

# Run, return "ct|decode_tok_s|wall_ms"
run_one() {
    local kind="$1" model="$2" port="$3"
    local body=$(jq -nc --arg m "$model" --arg p "$PROMPT" --argjson mt "$MAX_TOKENS" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    local t0 t1 resp
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    resp=$(curl -sf -m 240 -X POST "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    if [[ -z "$resp" ]]; then echo "ERR|0|0"; return; fi
    python3 - "$resp" "$t1" "$t0" <<'PY'
import json, sys
try: r = json.loads(sys.argv[1])
except: print("ERR|0|0"); sys.exit(0)
wall = int(sys.argv[2]) - int(sys.argv[3])
u = r.get('usage', {}) or {}
t = r.get('timings', {}) or {}
ct = u.get('completion_tokens', 0)
# Prefer server-side decode timing (mlx-serve only); else wall-based.
predicted_ms = t.get('predicted_ms', 0)
if predicted_ms > 0:
    dec = ct / (predicted_ms / 1000.0) if ct > 0 else 0
else:
    dec = ct / (wall / 1000.0) if (wall > 0 and ct > 0) else 0
print(f"{ct}|{dec:.1f}|{wall}")
PY
}

emit() {
    echo "$1|$2|$3|$4|$5|$6|${HW_TAG}" | tee -a "$OUT_CSV"
}

bench_pair() {
    local label="$1" kind="$2" model_id="$3"
    echo "==> $label / $kind" >&2
    local port
    case "$kind" in
        mlx-serve) start_mlx "$model_id" || return; port="$PORT_SERVE" ;;
        lmstudio)  start_lms "$model_id" || return; port="$PORT_LMS" ;;
    esac
    for run in $(seq 1 "$RUNS"); do
        IFS='|' read -r ct dec wall < <(run_one "$kind" "$model_id" "$port")
        echo "  run $run: ct=$ct decode=$dec tok/s wall=$wall ms" >&2
        emit "$label" "$kind" "$run" "$ct" "$dec" "$wall"
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
