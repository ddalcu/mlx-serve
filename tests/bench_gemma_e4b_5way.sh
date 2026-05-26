#!/bin/bash
# bench_gemma_e4b_5way.sh — Gemma 4 E4B across 5 engine × format combos.
#
#   1. mlx-serve  + MLX-4bit          (this repo, native Zig)
#   2. mlx-serve  + GGUF Q4_K_M       (this repo, embedded llama.cpp)
#   3. LM Studio  + MLX-4bit          (port 1234, gemma-4-e4b-it-mlx)
#   4. LM Studio  + GGUF Q4_K_M       (port 1234, gemma-4-e4b-it)
#   5. mlx-lm     + MLX-4bit          (CLI — mlx_lm.server crashes on
#                                      gemma4 with a Stream error)
#
# Methodology — wall-clock, uniform across engines:
#   For each HTTP config:
#     * 5 identical short requests (max_tokens=8) on the same prompt.
#       Run 1 = COLD TTFT, runs 2-5 averaged = WARM TTFT.
#     * 1 long request (max_tokens=128) on the same prompt (warm).
#       decode_tps = (128 - 8) / (wall_long - warm_wall_median) * 1000
#   For mlx-lm CLI:
#     * Each invocation is a fresh process (cold-only). The CLI itself
#       prints prompt + generation tokens-per-second on stdout; we use
#       those directly so the row is comparable for cold TTFT and decode.
#
# Output:
#   docs/perf-csvs/gemma_e4b_5way-YYYYMMDD-HHMM.csv
#   Columns: engine|format|metric|value_ms_or_tps|prompt_n|hw
#   Metrics emitted per config:
#     ttft_cold_ms       — wall clock for the first (cold) short request
#     ttft_warm_median_ms — median of warm short requests
#     decode_tps         — derived tokens/sec
#     prefill_tps        — prompt_n * 1000 / cold prompt_ms (server-side
#                          when available; else derived from wall)
set -uo pipefail

OUT_CSV="${1:-docs/perf-csvs/gemma_e4b_5way-$(date +%Y%m%d-%H%M).csv}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
PORT_SERVE=19200
PORT_LMS=1234
CTX=8192
WARMUP_RUNS=5
LONG_TOKENS=128
SHORT_TOKENS=8

MLX_PATH="/Volumes/Sandisk_1TB/Models/lmstudio-community/gemma-4-E4B-it-MLX-4bit"
GGUF_PATH="/Volumes/Sandisk_1TB/Models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
LMS_MLX_ID="gemma-4-e4b-it-mlx"
LMS_GGUF_ID="gemma-4-e4b-it"

mkdir -p "$(dirname "$OUT_CSV")"
[[ -x "$BIN" ]] || { echo "build mlx-serve first" >&2; exit 1; }
[[ -d "$MLX_PATH" ]] || { echo "MLX model missing: $MLX_PATH" >&2; exit 1; }
[[ -f "$GGUF_PATH" ]] || { echo "GGUF missing: $GGUF_PATH" >&2; exit 1; }
command -v jq >/dev/null
command -v mlx_lm.generate >/dev/null || { echo "mlx_lm CLI missing" >&2; exit 1; }
curl -sf --max-time 2 "http://127.0.0.1:$PORT_LMS/v1/models" 2>/dev/null | grep -q "$LMS_MLX_ID" \
    || { echo "LM Studio not running with $LMS_MLX_ID loaded" >&2; exit 1; }

HW="$(sysctl -n machdep.cpu.brand_string | tr ' ' '-')"
RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
HW_TAG="${HW}-${RAM_GB}gb"

echo "engine|format|metric|value|prompt_n|hw" > "$OUT_CSV"

PROMPT="$(python3 - <<'PY'
para = ("The kingdom of Avalon was beset by trials. Each season brought new "
        "challenges to its people, but the king remained steadfast. ")
print("Continue the story in 2 short sentences.\n\nBackground:\n" + para * 50)
PY
)"

emit() {
    local engine="$1" fmt="$2" metric="$3" value="$4" pn="${5:-0}"
    printf "%s|%s|%s|%s|%s|%s\n" "$engine" "$fmt" "$metric" "$value" "$pn" "$HW_TAG" | tee -a "$OUT_CSV"
}

SERVER_PID=""
cleanup_server() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 -f "mlx-serve.*port $PORT_SERVE" 2>/dev/null
    pkill -9 -f "mlx_lm" 2>/dev/null
    SERVER_PID=""
    sleep 1
}
trap cleanup_server EXIT INT TERM

start_mlx_serve() {
    local model="$1"
    cleanup_server
    "$BIN" --model "$model" --serve --port "$PORT_SERVE" --ctx-size "$CTX" \
        --prefix-cache-entries 4 --tokenize-cache-entries 4 \
        --llama-cache-entries 2 --log-level warn > /tmp/srv.log 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -sf --max-time 2 "http://127.0.0.1:$PORT_SERVE/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -20 /tmp/srv.log >&2; return 1; }
        sleep 0.5
    done
    return 1
}

# Returns: prompt_n|cached_n|prompt_ms|wall_ms|completion_tokens
http_ask() {
    local port="$1" body="$2"
    local t0 t1 resp
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    resp=$(curl -sf -m 240 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body")
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    [[ -z "$resp" ]] && { echo "0|0|0|0|0"; return; }
    python3 - "$resp" "$t1" "$t0" <<'PY'
import json, sys
try:
    r = json.loads(sys.argv[1])
except Exception:
    print("0|0|0|0|0"); sys.exit(0)
wall = int(sys.argv[2]) - int(sys.argv[3])
u = r.get('usage', {}) or {}
t = r.get('timings', {}) or {}
prompt_n = u.get('prompt_tokens', t.get('prompt_n', 0)) or 0
completion_n = u.get('completion_tokens', 0)
cached_n = t.get('cached_n', 0) or 0
prompt_ms = t.get('prompt_ms', 0) or 0
print(f"{prompt_n}|{cached_n}|{prompt_ms}|{wall}|{completion_n}")
PY
}

median() {
    python3 -c 'import sys, statistics; print(f"{statistics.median([float(x) for x in sys.stdin.read().split()]):.3f}")'
}

bench_http() {
    local engine="$1" fmt="$2" port="$3" model_id="$4"
    echo "==> $engine + $fmt" >&2
    local short_body long_body
    short_body=$(jq -nc --arg m "$model_id" --arg p "$PROMPT" --argjson t "$SHORT_TOKENS" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$t,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')
    long_body=$(jq -nc --arg m "$model_id" --arg p "$PROMPT" --argjson t "$LONG_TOKENS" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$t,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')

    local cold_pn=0 cold_wall=0 cold_pms=0
    local warm_walls=""
    local r line pn cn pms wall ct
    for r in $(seq 1 "$WARMUP_RUNS"); do
        line="$(http_ask "$port" "$short_body")"
        IFS='|' read -r pn cn pms wall ct <<< "$line"
        echo "  short r$r: prompt_n=$pn cached_n=$cn prompt_ms=$pms wall=$wall ct=$ct" >&2
        if [[ $r -eq 1 ]]; then
            cold_pn=$pn; cold_wall=$wall; cold_pms=$pms
        else
            warm_walls+="$wall "
        fi
    done
    line="$(http_ask "$port" "$long_body")"
    IFS='|' read -r long_pn long_cn long_pms long_wall long_ct <<< "$line"
    echo "  long:   prompt_n=$long_pn cached_n=$long_cn prompt_ms=$long_pms wall=$long_wall ct=$long_ct" >&2

    local warm_med
    warm_med=$(echo "$warm_walls" | median)
    emit "$engine" "$fmt" "ttft_cold_ms"        "$cold_wall" "$cold_pn"
    emit "$engine" "$fmt" "ttft_warm_median_ms" "$warm_med"  "$cold_pn"
    # decode_tps from long - warm_median, accounting for the 8 tokens of
    # decode already inside the warm wall.
    local decode_tps
    decode_tps=$(python3 -c "
long_wall, warm_med, lt, st, lct, sct = $long_wall, $warm_med, $LONG_TOKENS, $SHORT_TOKENS, $long_ct, $cold_pn
# Use long_ct (actual returned tokens) vs SHORT_TOKENS for the diff.
ct_diff = $long_ct - $SHORT_TOKENS
ms_diff = long_wall - warm_med
if ms_diff > 0 and ct_diff > 0:
    print(f'{ct_diff * 1000.0 / ms_diff:.2f}')
else:
    print('0')
")
    emit "$engine" "$fmt" "decode_tps" "$decode_tps" "$cold_pn"
    # Server-side prefill tps (cold). If server didn't emit prompt_ms,
    # fall back to wall-cold * (prompt_n / prompt_n+8) — rough.
    local prefill_tps
    if [[ "$cold_pms" != "0" ]] && [[ "$cold_pms" != "0.000" ]]; then
        prefill_tps=$(python3 -c "print(f'{$cold_pn * 1000.0 / $cold_pms:.2f}')")
    else
        # Subtract 8 tokens of decode from cold_wall (use decode_tps just
        # derived as the rate) — rough but honest. If decode_tps is 0
        # (couldn't be derived) just report wall as prompt time.
        prefill_tps=$(python3 -c "
cold_wall, dec_tps, st, cold_pn = $cold_wall, $decode_tps, $SHORT_TOKENS, $cold_pn
import sys
decode_ms = (st / dec_tps * 1000.0) if dec_tps > 0 else 0
prompt_ms = max(cold_wall - decode_ms, 1)
print(f'{cold_pn * 1000.0 / prompt_ms:.2f}')
")
    fi
    emit "$engine" "$fmt" "prefill_tps_cold" "$prefill_tps" "$cold_pn"
}

bench_mlxlm() {
    echo "==> mlx-lm + MLX-4bit (CLI, cold-only)" >&2
    local stdout_file stderr_file
    stdout_file="$(mktemp)"; stderr_file="$(mktemp)"
    mlx_lm.generate --model "$MLX_PATH" \
        --prompt "$PROMPT" --max-tokens "$LONG_TOKENS" --temp 0 \
        --extra-eos-token '<end_of_turn>' \
        > "$stdout_file" 2> "$stderr_file"
    # mlx_lm.generate prints metrics on STDOUT. Look for the lines.
    local prompt_n prompt_tps decode_tps
    prompt_n=$(awk '/^Prompt: / {gsub(",",""); print $2}' "$stdout_file")
    prompt_tps=$(awk '/^Prompt: / {print $(NF-1)}' "$stdout_file")
    decode_tps=$(awk '/^Generation: / {print $(NF-1)}' "$stdout_file")
    prompt_n="${prompt_n:-0}"; prompt_tps="${prompt_tps:-0}"; decode_tps="${decode_tps:-0}"
    echo "  prompt_n=$prompt_n prompt_tps=$prompt_tps decode_tps=$decode_tps" >&2

    local cold_ttft_ms
    cold_ttft_ms=$(python3 -c "
p = $prompt_n; tps = $prompt_tps; d = $decode_tps; st = $SHORT_TOKENS
prefill_ms = (p / tps * 1000.0) if tps > 0 else 0
decode_ms = (st / d * 1000.0) if d > 0 else 0
print(f'{prefill_ms + decode_ms:.0f}')
")
    emit "mlx-lm" "MLX-4bit" "ttft_cold_ms"        "$cold_ttft_ms" "$prompt_n"
    emit "mlx-lm" "MLX-4bit" "ttft_warm_median_ms" "0"              "$prompt_n"  # CLI = cold-only
    emit "mlx-lm" "MLX-4bit" "decode_tps"          "$decode_tps"    "$prompt_n"
    emit "mlx-lm" "MLX-4bit" "prefill_tps_cold"    "$prompt_tps"    "$prompt_n"
    rm -f "$stdout_file" "$stderr_file"
}

# ── ladder ──
start_mlx_serve "$MLX_PATH" || exit 1
bench_http "mlx-serve" "MLX-4bit" "$PORT_SERVE" "mlx-serve"
cleanup_server

start_mlx_serve "$GGUF_PATH" || exit 1
bench_http "mlx-serve" "GGUF-Q4_K_M" "$PORT_SERVE" "mlx-serve"
cleanup_server

bench_mlxlm

echo "==> warming up LM Studio MLX" >&2
curl -sf -m 600 -X POST "http://127.0.0.1:$PORT_LMS/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$LMS_MLX_ID" '{model:$m,messages:[{role:"user",content:"hi"}],max_tokens:1}')" \
    > /dev/null
bench_http "lmstudio" "MLX-4bit" "$PORT_LMS" "$LMS_MLX_ID"

echo "==> warming up LM Studio GGUF" >&2
curl -sf -m 600 -X POST "http://127.0.0.1:$PORT_LMS/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$LMS_GGUF_ID" '{model:$m,messages:[{role:"user",content:"hi"}],max_tokens:1}')" \
    > /dev/null
bench_http "lmstudio" "GGUF-Q4_K_M" "$PORT_LMS" "$LMS_GGUF_ID"

echo
echo "CSV: $OUT_CSV"
echo
column -t -s '|' "$OUT_CSV"
