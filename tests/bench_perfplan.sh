#!/bin/bash
# bench_perfplan.sh — performance-plan apples-to-apples harness.
#
# Runs the four scenarios from performance-plan.md (Phase 0) across
# mlx-serve and LM Studio against the same models, captures server-side
# timing (`timings.prompt_per_second`, `timings.cached_n`, decode tok/s),
# and emits a CSV that the companion plot script renders to PNG.
#
# Scenarios per (model, engine):
#   1. cold_prefill   — fresh server, unique 1k-token prompt, max_tokens=4
#   2. multi_turn     — 5-turn conversation; measure turn 2..5 TTFT
#   3. long_doc_cold  — 8k-token document + short Q, cold
#   4. long_doc_warm  — same doc, different Q (cold for the Q, warm for the doc)
#   5. decode         — short prompt, max_tokens=512 (pure decode rate)
#
# Engines:
#   mlx-serve   — booted by this script on $SERVER_PORT
#   lmstudio    — assumed running on :1234 (we don't manage it)
#
# Output CSV columns:
#   engine|model|scenario|run|prompt_n|cached_n|prefill_tok_s|decode_tok_s|first_byte_ms|wall_ms|notes
#
# Usage:
#   ./tests/bench_perfplan.sh <csv_out> <model_engine_pairs...>
#
# Each pair is a `|`-separated entry of:
#   engine_kind|label|model_path_or_id
# where engine_kind ∈ {mlx-serve, lmstudio} and model_path_or_id is the
# filesystem path (mlx-serve) or the LM Studio model id (lmstudio).
#
# Example:
#   ./tests/bench_perfplan.sh docs/perf-csvs/baseline.csv \
#     "mlx-serve|qwen3.5-4b-mlx|/Volumes/Sandisk_1TB/Models/mlx-community/Qwen3.5-4B-MLX-4bit" \
#     "lmstudio|qwen3.5-4b-mlx|qwen3.5-4b-mlx"

set -uo pipefail

CSV_OUT="${1:-}"; shift || true
[[ -z "$CSV_OUT" || "$#" -eq 0 ]] && { echo "usage: $0 <csv_out> <engine|label|path>..." >&2; exit 1; }

BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
SERVER_PORT="${SERVER_PORT:-19044}"
LMS_PORT="${LMS_PORT:-1234}"
CTX="${CTX:-8192}"
RUNS_COLD="${RUNS_COLD:-3}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
LONG_DOC_TOKENS="${LONG_DOC_TOKENS:-6500}"  # roughly 8k after chat template

mkdir -p "$(dirname "$CSV_OUT")"
HW="$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr ' ' '-')"
RAM_GB=$(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024))
HW_TAG="${HW}-${RAM_GB}gb"

echo "engine|model|scenario|run|prompt_n|cached_n|prefill_tok_s|decode_tok_s|first_byte_ms|wall_ms|hw|notes" > "$CSV_OUT"

# ── Prompt builders (Python so we control token counts roughly) ──
make_prompt() { python3 - "$1" <<'PY'
import sys
n=int(sys.argv[1])
sent="The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
reps=max(1, n//13)
print("Summarize the following text in one word.\n\n" + sent*reps, end="")
PY
}

make_doc() { python3 - "$1" <<'PY'
import sys
n=int(sys.argv[1])
para=("The atmosphere of the Earth is composed of layers, each with distinct properties. "
      "The troposphere extends up to about 10 km and contains weather. The stratosphere extends "
      "to roughly 50 km and holds the ozone layer that absorbs ultraviolet radiation. ")
reps=max(1, n//40)
print("Below is a long document. Answer the question that follows.\n\nDOCUMENT:\n" + para*reps + "\n\nQUESTION: ", end="")
PY
}

# ── Engine lifecycle ──
SERVER_PID=""
SERVER_LOG=""
stop_mlxserve() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 mlx-serve 2>/dev/null
    sleep 1
    SERVER_PID=""
}

start_mlxserve() {
    local model="$1"
    stop_mlxserve
    SERVER_LOG="$(mktemp -t perfplan_server.XXXXXX)"
    "$BINARY" --model "$model" --serve --port "$SERVER_PORT" \
        --ctx-size "$CTX" --prefix-cache-entries 8 --log-level warn \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -sf --max-time 2 "http://127.0.0.1:$SERVER_PORT/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "  mlx-serve died:" >&2; tail -20 "$SERVER_LOG" >&2; return 1; }
        sleep 0.5
    done
    echo "  mlx-serve never came up" >&2
    return 1
}

# LM Studio is assumed to be running. We just verify and JIT-load the model.
verify_lms() {
    curl -sf "http://127.0.0.1:$LMS_PORT/v1/models" >/dev/null || {
        echo "  LM Studio not on :$LMS_PORT — start it via 'lms server start'" >&2
        return 1
    }
}

lms_jit_load() {
    local model_id="$1"
    local body
    body=$(jq -nc --arg m "$model_id" '{model:$m,messages:[{role:"user",content:"hi"}],max_tokens:1,stream:false}')
    curl -sf -m 600 -X POST "http://127.0.0.1:$LMS_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body" >/dev/null
}

lms_unload_all() {
    lms unload --all >/dev/null 2>&1 || true
    sleep 2
}

# ── Request helpers (engine-agnostic) ──
# send_one engine model prompt max_tokens stream → "wall_ms|first_byte_ms|prompt_n|cached_n|completion_n|prefill_ms|decode_ms"
send_one() {
    local engine="$1" model="$2" prompt="$3" mt="$4" stream="$5"
    local port body
    case "$engine" in
        mlx-serve) port="$SERVER_PORT" ;;
        lmstudio)  port="$LMS_PORT" ;;
        *) echo "ERR|0|0|0|0|0|0"; return ;;
    esac
    body=$(jq -nc --arg p "$prompt" --arg m "$model" --argjson mt "$mt" --argjson stream "$stream" \
        '{model:$m,messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,stream:$stream}')

    local t0 t1 fb resp tmp
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    if [[ "$stream" == "true" ]]; then
        tmp="$(mktemp)"
        # Stream + capture first-byte time. curl --no-buffer + write each chunk to file; tag first chunk arrival.
        curl -sN -m 240 -H "Content-Type: application/json" -d "$body" \
            "http://127.0.0.1:$port/v1/chat/completions" > "$tmp" 2>/dev/null &
        local cpid=$!
        fb=0
        while kill -0 "$cpid" 2>/dev/null; do
            if [[ -s "$tmp" ]]; then
                fb=$(python3 -c 'import time;print(int(time.time()*1000))')
                fb=$((fb - t0))
                break
            fi
            sleep 0.005
        done
        wait "$cpid" 2>/dev/null
        t1=$(python3 -c 'import time;print(int(time.time()*1000))')
        # Stream doesn't easily give us server timings — just measure wall + first byte.
        # Estimate completion tokens via line count of data: chunks containing content deltas.
        local ct
        ct=$(grep -c '"delta":{"content":' "$tmp" 2>/dev/null || echo 0)
        rm -f "$tmp"
        echo "$((t1-t0))|${fb}|0|0|${ct}|0|0"
        return
    fi

    resp=$(curl -sf -m 240 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$body" 2>/dev/null)
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    if [[ -z "$resp" ]]; then echo "ERR|0|0|0|0|0|0"; return; fi
    python3 - "$resp" "$t1" "$t0" <<'PY'
import json,sys
try:
    r=json.loads(sys.argv[1])
except Exception:
    print("ERR|0|0|0|0|0|0"); sys.exit(0)
wall=int(sys.argv[2])-int(sys.argv[3])
u=r.get('usage',{}) or {}
t=r.get('timings',{}) or {}
print(f"{wall}|0|{u.get('prompt_tokens',0)}|{t.get('cached_n',0)}|{u.get('completion_tokens',0)}|{t.get('prompt_ms',0)}|{t.get('predicted_ms',0)}")
PY
}

emit() {
    local engine="$1" model="$2" scenario="$3" run="$4" pn="$5" cn="$6" pf="$7" dc="$8" fb="$9" wall="${10}" notes="${11:-}"
    echo "${engine}|${model}|${scenario}|${run}|${pn}|${cn}|${pf}|${dc}|${fb}|${wall}|${HW_TAG}|${notes}" | tee -a "$CSV_OUT"
}

# ── Scenarios ──
run_cold_prefill() {
    local engine="$1" model="$2" model_id="$3"
    local p; p="$(make_prompt 1000)"
    for r in $(seq 1 "$RUNS_COLD"); do
        # restart engine each run for true cold
        case "$engine" in
            mlx-serve) start_mlxserve "$model" || return ;;
            lmstudio)  lms_unload_all; lms_jit_load "$model_id" || return ;;
        esac
        local nonce="run-$r-$RANDOM unique preamble. "
        IFS='|' read -r wall fb pn cn ct pms dms < <(send_one "$engine" "$model_id" "${nonce}${p}" 4 false)
        local pf=0 dc=0
        if [[ "$pms" != "0" && "$pn" != "0" ]]; then
            pf=$(python3 -c "p=$pn;c=$cn;t=$pms;print(f'{(p-c)/(t/1000):.1f}' if t>0 else '0')")
        elif [[ "$wall" != "0" && "$pn" != "0" ]]; then
            pf=$(python3 -c "p=$pn;w=$wall;print(f'{p/(w/1000):.1f}' if w>0 else '0')")
        fi
        emit "$engine" "$model_id" "cold_prefill" "$r" "$pn" "$cn" "$pf" "$dc" "$fb" "$wall" ""
    done
}

run_multi_turn() {
    local engine="$1" model="$2" model_id="$3"
    # Single engine instance; 5 turns. We don't have a multi-message body builder
    # here (would need history accumulation), so we approximate "warm" by sending
    # the same long preamble + a varying suffix on each turn — the prefix cache
    # should reuse the shared preamble.
    case "$engine" in
        mlx-serve) start_mlxserve "$model" || return ;;
        lmstudio)  lms_unload_all; lms_jit_load "$model_id" || return ;;
    esac
    local preamble; preamble="$(make_prompt 1000)"
    for r in $(seq 1 5); do
        local prompt="${preamble} | Turn $r: tell me about topic ${r}."
        IFS='|' read -r wall fb pn cn ct pms dms < <(send_one "$engine" "$model_id" "$prompt" 16 true)
        local dc=0
        # For streaming, server timings aren't in the response — use wall-clock
        # over completion tokens (approximate).
        if [[ "$ct" -gt 0 && "$wall" -gt 0 ]]; then
            dc=$(python3 -c "c=$ct;w=$wall;print(f'{c/(w/1000):.1f}')")
        fi
        emit "$engine" "$model_id" "multi_turn_turn${r}" "$r" "$pn" "$cn" "0" "$dc" "$fb" "$wall" "streaming"
    done
}

run_long_doc() {
    local engine="$1" model="$2" model_id="$3"
    local doc; doc="$(make_doc "$LONG_DOC_TOKENS")"

    # Cold: fresh engine, unique doc nonce, ask Q1
    case "$engine" in
        mlx-serve) start_mlxserve "$model" || return ;;
        lmstudio)  lms_unload_all; lms_jit_load "$model_id" || return ;;
    esac
    local q1="What does the document say about the troposphere? Short answer please."
    local nonce="doc-bench-$RANDOM "
    IFS='|' read -r wall fb pn cn ct pms dms < <(send_one "$engine" "$model_id" "${nonce}${doc}${q1}" 32 false)
    local pf=0
    [[ "$pms" != "0" && "$pn" != "0" ]] && pf=$(python3 -c "p=$pn;c=$cn;t=$pms;print(f'{(p-c)/(t/1000):.1f}' if t>0 else '0')")
    emit "$engine" "$model_id" "long_doc_cold" "1" "$pn" "$cn" "$pf" "0" "$fb" "$wall" ""

    # Warm: same doc, different question
    local q2="And what does it say about the stratosphere? Short answer."
    IFS='|' read -r wall fb pn cn ct pms dms < <(send_one "$engine" "$model_id" "${nonce}${doc}${q2}" 32 false)
    pf=0
    [[ "$pms" != "0" && "$pn" != "0" ]] && pf=$(python3 -c "p=$pn;c=$cn;t=$pms;print(f'{(p-c)/(t/1000):.1f}' if t>0 else '0')")
    emit "$engine" "$model_id" "long_doc_warm" "1" "$pn" "$cn" "$pf" "0" "$fb" "$wall" ""
}

run_decode() {
    local engine="$1" model="$2" model_id="$3"
    case "$engine" in
        mlx-serve) start_mlxserve "$model" || return ;;
        lmstudio)  lms_unload_all; lms_jit_load "$model_id" || return ;;
    esac
    # Short prompt, max_tokens=DECODE_TOKENS. Server timings give us predicted_per_second.
    local prompt="Write a detailed essay about quantum computing"
    for r in 1 2; do
        # 1st is warmup, 2nd is the measurement
        IFS='|' read -r wall fb pn cn ct pms dms < <(send_one "$engine" "$model_id" "$prompt" "$DECODE_TOKENS" false)
        if [[ "$r" == "2" ]]; then
            local dc=0
            if [[ "$dms" != "0" && "$ct" != "0" ]]; then
                dc=$(python3 -c "c=$ct;t=$dms;print(f'{c/(t/1000):.1f}' if t>0 else '0')")
            elif [[ "$wall" -gt 0 && "$ct" -gt 0 ]]; then
                dc=$(python3 -c "c=$ct;w=$wall;print(f'{c/(w/1000):.1f}')")
            fi
            emit "$engine" "$model_id" "decode" "$r" "$pn" "$cn" "0" "$dc" "$fb" "$wall" ""
        fi
    done
}

# ── Pre-flight ──
[[ -x "$BINARY" ]] || { echo "Build first: zig build -Doptimize=ReleaseFast" >&2; exit 1; }
command -v jq >/dev/null   || { echo "needs jq" >&2; exit 1; }

trap 'stop_mlxserve; rm -f "$SERVER_LOG"; true' EXIT

# ── Run the matrix ──
for pair in "$@"; do
    IFS='|' read -r engine label path_or_id <<<"$pair"
    echo "=== ${engine} / ${label} ===" >&2
    case "$engine" in
        lmstudio) verify_lms || continue ;;
    esac
    run_cold_prefill "$engine" "$path_or_id" "$label" || true
    run_multi_turn   "$engine" "$path_or_id" "$label" || true
    run_long_doc     "$engine" "$path_or_id" "$label" || true
    run_decode       "$engine" "$path_or_id" "$label" || true
done

stop_mlxserve
echo
echo "CSV: $CSV_OUT"
