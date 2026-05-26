#!/bin/bash
# bench_iter2to5.sh — Iteration 8 bench: shows the warm wins from this
# overnight push without requiring LM Studio. Captures:
#   - Cold TTFT for a 1.8k-token Gemma 4 MLX prompt (was 2880 ms before
#     Iteration 2, expected 2880 ms still — Metal prefill unchanged).
#   - Warm TTFT for the same prompt (was ~271 ms before Iteration 2,
#     expected 35 ms after — the tokenize cache strips the ~236 ms
#     render+tokenize round-trip).
#   - GGUF alternating-prompt cached_n with --llama-cache-entries 2 (was
#     ~3 before Iteration 3-5; expected ~71 — full B-cycle survives).
#
# Output: docs/perf-csvs/bench_iter2to5-YYYYMMDD-HHMM.csv
set -uo pipefail

OUT_CSV="${1:-docs/perf-csvs/bench_iter2to5-$(date +%Y%m%d-%H%M).csv}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
PORT=19099
CTX=8192

mkdir -p "$(dirname "$OUT_CSV")"
[[ -x "$BIN" ]] || { echo "build mlx-serve first" >&2; exit 1; }
command -v jq >/dev/null

HW="$(sysctl -n machdep.cpu.brand_string | tr ' ' '-')"
RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
HW_TAG="${HW}-${RAM_GB}gb"

echo "label|run|prompt_n|cached_n|tokenize_ms|prompt_ms|predicted_per_second|hw" > "$OUT_CSV"

GEMMA="/Volumes/Sandisk_1TB/Models/mlx-community/gemma-4-e4b-it-4bit"
QWEN_GGUF="/Volumes/Sandisk_1TB/Models/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-IQ4_NL.gguf"

SERVER_PID=""
cleanup() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    pkill -9 mlx-serve 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT INT TERM

start_mlx() {
    local args="$@"
    cleanup
    "$BIN" --serve --port "$PORT" --ctx-size "$CTX" --log-level warn $args > /tmp/bench.log 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "  server died" >&2; tail -20 /tmp/bench.log >&2; return 1; }
        sleep 0.5
    done
    return 1
}

ask() {
    local prompt="$1" max="${2:-8}"
    local body=$(jq -nc --arg p "$prompt" --argjson m "$max" \
        '{messages:[{role:"user",content:$p}],max_tokens:$m,temperature:0,stream:false}')
    curl -sf -m 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body"
}

emit_response() {
    local label="$1" run="$2" resp="$3"
    python3 - "$label" "$run" "$resp" "$HW_TAG" <<'PY' >> "$OUT_CSV"
import json, sys
label = sys.argv[1]; run = sys.argv[2]
try:
    r = json.loads(sys.argv[3])
except Exception:
    print(f"{label}|{run}|ERR|0|0|0|0|{sys.argv[4]}"); sys.exit(0)
t = r.get('timings', {}) or {}
print(f"{label}|{run}|{t.get('prompt_n',0)}|{t.get('cached_n',0)}|{t.get('tokenize_ms',0)}|{t.get('prompt_ms',0)}|{t.get('predicted_per_second',0)}|{sys.argv[4]}")
PY
}

# ── Iteration 2 win: long-prompt warm reuse (MLX Gemma 4 E4B) ──
echo "==> gemma4-e4b MLX long-prompt warm" >&2
if [[ -d "$GEMMA" ]]; then
    start_mlx --model "$GEMMA" --prefix-cache-entries 4 --tokenize-cache-entries 4 || exit 1
    LONG=$(python3 -c "print('Continue the story: ' + ('The kingdom of Avalon was beset by trials. ' * 200))")
    for r in 1 2 3 4 5; do
        RESP="$(ask "$LONG" 8)"
        echo "  run $r: $(echo $RESP | jq -c '.timings')" >&2
        emit_response "gemma4-e4b-mlx" "$r" "$RESP"
    done
fi

# ── Iteration 3-5 win: alternating-prompt LRU (GGUF Qwen3.5-4B) ──
if [[ -f "$QWEN_GGUF" ]]; then
    echo "==> qwen35-4b-gguf alternating with --llama-cache-entries 2" >&2
    start_mlx --model "$QWEN_GGUF" --llama-cache-entries 2 --tokenize-cache-entries 4 || exit 1
    PA="The first chapter describes a sunny morning where the protagonist wakes up in a small village by the sea. Birds chirp loudly. Summarize."
    PB="In the dense jungle a tiger stalks its prey at twilight. The leaves rustle as a herd of deer passes by. Summarize."
    r=0
    for p in "$PA" "$PB" "$PA" "$PB" "$PA"; do
        r=$((r+1))
        if [[ "$p" == "$PA" ]]; then lbl="qwen35-4b-gguf-multi-A"; else lbl="qwen35-4b-gguf-multi-B"; fi
        RESP="$(ask "$p" 8)"
        echo "  run $r ($lbl): $(echo $RESP | jq -c '.timings')" >&2
        emit_response "$lbl" "$r" "$RESP"
    done

    echo "==> qwen35-4b-gguf same-pattern with --llama-cache-entries 1 (regression check)" >&2
    start_mlx --model "$QWEN_GGUF" --llama-cache-entries 1 --tokenize-cache-entries 4 || exit 1
    r=0
    for p in "$PA" "$PB" "$PA" "$PB" "$PA"; do
        r=$((r+1))
        if [[ "$p" == "$PA" ]]; then lbl="qwen35-4b-gguf-single-A"; else lbl="qwen35-4b-gguf-single-B"; fi
        RESP="$(ask "$p" 8)"
        echo "  run $r ($lbl): $(echo $RESP | jq -c '.timings')" >&2
        emit_response "$lbl" "$r" "$RESP"
    done
fi

cleanup
echo
echo "CSV: $OUT_CSV"
echo
echo "--- summary ---"
column -t -s '|' "$OUT_CSV"
