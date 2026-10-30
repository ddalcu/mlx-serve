#!/usr/bin/env bash
# test_load_progress.sh — GET /v1/load-progress: a headless server lists nothing; a GPU load is seen mid
# `reading_weights` (bytes) and a Neural Engine load mid `ane_programs` (programs), each with a total and a count that
# climbs, and the list empties once ready; `/props` registry_generation moves with the load; a POST naming a model
# loads nothing; a `--model` boot answers /health 503 `loading` and the progress while it loads.
#
#   ./tests/test_load_progress.sh [port]   # LOAD_PROGRESS_MODEL (Qwen3.8-27B, else Qwen3.5-4B),
#                                          # LOAD_PROGRESS_ANE_MODEL (Qwen3.5-0.8B)
#
# Scratch HOME (the ANE opt-in goes in its model-settings.json); the ANE compile cache stays the real one. SKIPs
# without the packs; the ANE half without the private framework. Hermetic twin: host/registry/load_progress.zig tests.

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11387}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
source "$ROOT/tests/_lib_models.sh"
GPU_MODEL="${LOAD_PROGRESS_MODEL:-$(find_model ddalcu/Qwen3.8-27B-MLX-Serve-4bit mlx-community/Qwen3.5-4B-MLX-4bit)}"
ANE_MODEL="${LOAD_PROGRESS_ANE_MODEL:-$(find_model mlx-community/Qwen3.5-0.8B-MLX-4bit)}"
[ -n "$GPU_MODEL" ] || { echo "SKIP: needs a pack (LOAD_PROGRESS_MODEL)"; exit 0; }

PASS=0; FAIL=0
check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi
}
FAKE_HOME="$(mktemp -d)"
mkdir -p "$FAKE_HOME/.mlx-serve"
export MLX_SERVE_ANE_CACHE_DIR="${MLX_SERVE_ANE_CACHE_DIR:-$HOME/.mlx-serve/ane-cache}"
LOG="$FAKE_HOME/server.log"
SRV=""
stop() {
    [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    for _ in $(seq 1 40); do pgrep -f "mlx-serve.*--port $PORT" >/dev/null || break; sleep 0.5; done
    SRV=""
}
cleanup() { stop; rm -rf "$FAKE_HOME"; }
trap cleanup EXIT
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null

boot() { # boot <model dir root>: headless, the root discovered
    : >"$LOG"
    HOME="$FAKE_HOME" "$BIN" --serve --model-dir "$1" --host 127.0.0.1 --port "$PORT" --log-level info --log-file off >"$LOG" 2>&1 &
    SRV=$!
    srv_wait_health "http://127.0.0.1:$PORT" 240 0.5 "$SRV" && return 0
    echo "FAIL: server never became healthy"; tail -5 "$LOG"; exit 1
}
# watch <model path> <out>: load it in the background, recording /v1/load-progress every 50 ms.
watch() {
    : >"$2"
    curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/load-model" -H 'Content-Type: application/json' -d "{\"model\":\"$1\"}" >/dev/null &
    local loader=$!
    while kill -0 "$loader" 2>/dev/null; do
        curl -s "http://127.0.0.1:$PORT/v1/load-progress" >>"$2"; echo >>"$2"
        sleep 0.05
    done
    curl -s "http://127.0.0.1:$PORT/v1/load-progress" >"$2.after"
}
# seen <file> <phase> <field>: the largest value of progress.<field> while in <phase>.
seen() {
    python3 -c "
import sys, json
best = -1
for line in open(sys.argv[1]):
    line = line.strip()
    if not line: continue
    for m in json.loads(line).get('loading', []):
        p = m['progress']
        if p['phase'] == sys.argv[2]: best = max(best, p[sys.argv[3]])
print(best)" "$1" "$2" "$3"
}

echo "[1] headless: an empty list"
boot "$(dirname "$(dirname "$GPU_MODEL")")"
R="$(curl -s "http://127.0.0.1:$PORT/v1/load-progress")"
check "[1] {\"loading\":[]} (got $R)" "$([ "$R" = '{"loading":[]}' ] && echo 1 || echo 0)"
generation() { curl -s "http://127.0.0.1:$PORT/props" | python3 -c "import sys, json; print(json.load(sys.stdin).get('registry_generation', -1))"; }
G0="$(generation)"
check "[1] /props registry_generation (got $G0)" "$([ "${G0:--1}" -ge 0 ] && echo 1 || echo 0)"
R="$(curl -s -X POST "http://127.0.0.1:$PORT/v1/load-progress" -H 'Content-Type: application/json' -d "{\"model\":\"$GPU_MODEL\"}")"
check "[1] a POST naming a model loads nothing (got $R)" "$([ "$R" = '{"loading":[]}' ] && [ "$(generation)" = "$G0" ] && echo 1 || echo 0)"

echo "[2] a GPU load reads weights ($GPU_MODEL)"
watch "$GPU_MODEL" "$FAKE_HOME/gpu.jsonl"
TOTAL="$(seen "$FAKE_HOME/gpu.jsonl" reading_weights total)"
DONE="$(seen "$FAKE_HOME/gpu.jsonl" reading_weights done)"
check "[2] reading_weights with a total (got $TOTAL bytes)" "$([ "${TOTAL:--1}" -gt 0 ] && echo 1 || echo 0)"
check "[2] bytes read climb (got $DONE)" "$([ "${DONE:--1}" -gt 0 ] && echo 1 || echo 0)"
check "[2] empty once ready" "$([ "$(cat "$FAKE_HOME/gpu.jsonl.after")" = '{"loading":[]}' ] && echo 1 || echo 0)"
G1="$(generation)"
check "[2] registry_generation moved with the load ($G0 -> $G1)" "$([ "${G1:--1}" -gt "${G0:-0}" ] && echo 1 || echo 0)"
stop

if [ -n "$ANE_MODEL" ] && [ -d /System/Library/PrivateFrameworks/AppleNeuralEngine.framework ]; then
    echo "[3] a Neural Engine load builds programs ($ANE_MODEL)"
    printf '{ "%s": { "ane": true } }\n' "$ANE_MODEL" >"$FAKE_HOME/.mlx-serve/model-settings.json"
    boot "$(dirname "$(dirname "$ANE_MODEL")")"
    watch "$ANE_MODEL" "$FAKE_HOME/ane.jsonl"
    TOTAL="$(seen "$FAKE_HOME/ane.jsonl" ane_programs total)"
    DONE="$(seen "$FAKE_HOME/ane.jsonl" ane_programs done)"
    check "[3] ane_programs with a total (got $TOTAL)" "$([ "${TOTAL:--1}" -gt 0 ] && echo 1 || echo 0)"
    check "[3] programs climb (got $DONE)" "$([ "${DONE:--1}" -gt 0 ] && echo 1 || echo 0)"
    check "[3] log: [ane] ready" "$(grep -q '\[ane\] ready' "$LOG" && echo 1 || echo 0)"
    check "[3] empty once ready" "$([ "$(cat "$FAKE_HOME/ane.jsonl.after")" = '{"loading":[]}' ] && echo 1 || echo 0)"
    stop
else
    echo "  SKIP [3]: no Neural Engine pack or framework"
fi

echo "[4] a --model boot answers while its model loads ($GPU_MODEL)"
: >"$LOG"
: >"$FAKE_HOME/boot.txt"
HOME="$FAKE_HOME" "$BIN" --serve --model "$GPU_MODEL" --host 127.0.0.1 --port "$PORT" --log-level info --log-file off >"$LOG" 2>&1 &
SRV=$!
CODE=""
for _ in $(seq 1 2400); do
    CODE="$(curl -s -o "$FAKE_HOME/h.json" -w '%{http_code}' "http://127.0.0.1:$PORT/health")"
    [ "$CODE" = "200" ] && break
    if [ "$CODE" = "503" ]; then
        { cat "$FAKE_HOME/h.json"; echo; curl -s "http://127.0.0.1:$PORT/v1/load-progress"; echo; } >>"$FAKE_HOME/boot.txt"
    fi
    kill -0 "$SRV" 2>/dev/null || break
    sleep 0.05
done
check "[4] /health is 503 loading during the startup load" "$(grep -q '"status":"loading"' "$FAKE_HOME/boot.txt" && echo 1 || echo 0)"
check "[4] /v1/load-progress names the loading model" "$(grep -q '"phase":"' "$FAKE_HOME/boot.txt" && echo 1 || echo 0)"
check "[4] healthy once loaded (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
stop

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
