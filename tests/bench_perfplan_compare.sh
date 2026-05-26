#!/bin/bash
# bench_perfplan_compare.sh — head-to-head mlx-serve vs LM Studio across the
# performance-plan.md scenarios. Captures server-side prefill timing (no
# guessing from wall-time), runs cold + warm scenarios, writes CSV + JSON.
#
# Usage:
#   ./tests/bench_perfplan_compare.sh <out_csv> <pair...>
#
# pair = "label|kind|path_or_id"
#   kind in {mlx-serve, lmstudio}
#   for mlx-serve: path is a model dir or .gguf
#   for lmstudio:  path_or_id is the LMS model id (run `lms ls`)
#
# Example:
#   ./tests/bench_perfplan_compare.sh docs/perf-csvs/post-phase1.csv \
#     "qwen35-4b-mlx|mlx-serve|/Volumes/Sandisk_1TB/Models/mlx-community/Qwen3.5-4B-MLX-4bit" \
#     "qwen35-4b-mlx|lmstudio|qwen3.5-4b-mlx" \
#     "gemma4-e4b|mlx-serve|/Users/david/.mlx-serve/models/gemma-4-e4b-it-4bit" \
#     "gemma4-e4b|lmstudio|gemma-4-e4b-it-mlx" \
#     "qwen35-4b-gguf|mlx-serve|/Volumes/Sandisk_1TB/Models/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-IQ4_NL.gguf" \
#     "qwen35-4b-gguf|lmstudio|qwen3.5-4b"
set -uo pipefail

OUT="${1:-}"; shift || true
[[ -z "$OUT" || "$#" -eq 0 ]] && { echo "usage: $0 <out_csv> <label|kind|path>..." >&2; exit 1; }

BIN="${BINARY:-./zig-out/bin/mlx-serve}"
PORT_SERVE="${PORT_SERVE:-19099}"
PORT_LMS="${PORT_LMS:-1234}"
CTX="${CTX:-8192}"
WARM_RUNS="${WARM_RUNS:-3}"

[[ -x "$BIN" ]] || { echo "build first" >&2; exit 1; }
command -v jq >/dev/null || { echo "needs jq" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

HW="$(sysctl -n machdep.cpu.brand_string | tr ' ' '-')"
RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
HW_TAG="${HW}-${RAM_GB}gb"

echo "label|kind|model|scenario|run|prompt_n|cached_n|prompt_ms|prompt_tok_s|wall_ms|hw|notes" > "$OUT"

emit() {
    local label="$1" kind="$2" model="$3" scenario="$4" run="$5"
    local pn="$6" cn="$7" pms="$8" tps="$9" wall="${10}" notes="${11:-}"
    echo "${label}|${kind}|${model}|${scenario}|${run}|${pn}|${cn}|${pms}|${tps}|${wall}|${HW_TAG}|${notes}" | tee -a "$OUT"
}

SERVER_PID=""
stop_mlx() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 mlx-serve 2>/dev/null
    sleep 1
}

start_mlx() {
    local path="$1"
    stop_mlx
    "$BIN" --model "$path" --serve --port "$PORT_SERVE" --ctx-size "$CTX" \
        --prefix-cache-entries 8 --log-level warn > /tmp/bench_compare.log 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -sf --max-time 2 "http://127.0.0.1:$PORT_SERVE/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "mlx-serve died:" >&2; tail -20 /tmp/bench_compare.log; return 1; }
        sleep 0.5
    done
    return 1
}

# LMS JIT-load via warmup curl.
lms_load() {
    local model="$1"
    lms unload --all >/dev/null 2>&1
    sleep 2
    local body=$(jq -nc --arg m "$model" '{model:$m,messages:[{role:"user",content:"hi"}],max_tokens:1,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    curl -sf -m 600 -X POST "http://127.0.0.1:$PORT_LMS/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" >/dev/null
}

# Request: returns "wall_ms|prompt_n|cached_n|prompt_ms|prompt_tok_s"
send_req() {
    local kind="$1" model="$2" prompt="$3" max_tokens="$4"
    local port body
    case "$kind" in
        mlx-serve) port="$PORT_SERVE" ;;
        lmstudio) port="$PORT_LMS" ;;
        *) echo "ERR|0|0|0|0"; return ;;
    esac
    body=$(jq -nc --arg m "$model" --arg p "$prompt" --argjson mt "$max_tokens" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    local t0 t1
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    local resp
    resp=$(curl -sf -m 240 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" 2>/dev/null)
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
pn = u.get('prompt_tokens', 0)
cn = t.get('cached_n', 0)
pms = t.get('prompt_ms', 0)
pps = t.get('prompt_per_second', 0)
print(f"{wall}|{pn}|{cn}|{pms}|{pps}")
PY
}

bench_cold() {
    local label="$1" kind="$2" path="$3"
    local prompt="$(python3 -c 'print("Summarize in one word.\n\n" + "The quick brown fox jumps over the lazy dog. " * 70)')"
    # cold: restart engine for each run so KV cache is fully fresh
    for run in $(seq 1 2); do
        case "$kind" in
            mlx-serve) start_mlx "$path" || return ;;
            lmstudio) lms_load "$path" || return ;;
        esac
        local salt="run-$run-$RANDOM-$RANDOM "
        IFS='|' read -r wall pn cn pms tps < <(send_req "$kind" "$(basename "$path")" "${salt}${prompt}" 4)
        # For lmstudio the model id is the path itself (not basename), retry if needed
        if [[ "$wall" == "ERR" && "$kind" == "lmstudio" ]]; then
            IFS='|' read -r wall pn cn pms tps < <(send_req "$kind" "$path" "${salt}${prompt}" 4)
        fi
        # If mlx-serve used a .gguf basename, retry with full path
        if [[ "$wall" == "ERR" && "$kind" == "mlx-serve" ]]; then
            IFS='|' read -r wall pn cn pms tps < <(send_req "$kind" "$path" "${salt}${prompt}" 4)
        fi
        emit "$label" "$kind" "$path" "cold" "$run" "$pn" "$cn" "$pms" "$tps" "$wall" ""
    done
}

bench_warm() {
    local label="$1" kind="$2" path="$3"
    local prompt="$(python3 -c 'print("Tell me a short story.\n\n" + "Background: The kingdom was at peace. " * 60)')"
    case "$kind" in
        mlx-serve) start_mlx "$path" || return ;;
        lmstudio) lms_load "$path" || return ;;
    esac
    # Run the same prompt N times: first is cold, subsequent are warm
    for run in $(seq 1 $((WARM_RUNS + 1))); do
        local model_id; model_id="$(basename "$path")"
        IFS='|' read -r wall pn cn pms tps < <(send_req "$kind" "$model_id" "$prompt" 4)
        if [[ "$wall" == "ERR" ]]; then
            IFS='|' read -r wall pn cn pms tps < <(send_req "$kind" "$path" "$prompt" 4)
        fi
        local scen=$([[ "$run" == "1" ]] && echo "warm_first" || echo "warm_repeat${run}")
        emit "$label" "$kind" "$path" "$scen" "$run" "$pn" "$cn" "$pms" "$tps" "$wall" ""
    done
}

trap 'stop_mlx; lms unload --all >/dev/null 2>&1; true' EXIT INT TERM

for pair in "$@"; do
    IFS='|' read -r label kind path <<<"$pair"
    echo "==> $label / $kind / $path" >&2
    case "$kind" in
        mlx-serve)
            [[ -e "$path" ]] || { echo "  SKIP: missing $path" >&2; continue; } ;;
        lmstudio)
            # Verify LM Studio is up
            curl -sf "http://127.0.0.1:$PORT_LMS/v1/models" >/dev/null || { echo "  SKIP: LMS not on :$PORT_LMS" >&2; continue; } ;;
    esac
    bench_cold "$label" "$kind" "$path"
    bench_warm "$label" "$kind" "$path"
done

stop_mlx
lms unload --all >/dev/null 2>&1 || true
echo
echo "CSV: $OUT"
