#!/bin/bash
# Usage:
#   PRESSURE_TEST_MODEL=<model-dir> MLX_SERVE_BINARY=<ReleaseFast-bin> \
#     bash tests/test_prefix_cache_pressure.sh [model-dir] [port]
#
# Phase 1 (default): starts mlx-serve with an isolated HOME, a small hot cache,
#   and MLX_SERVE_PRESSURE_OVERRIDE pointing at a level file initialized to normal;
#   sends distinct chat requests to fill the cache, switches the override file to
#   warning, and asserts B3's budget-step log line:
#   `[hot-cache] memory pressure worse: (warning|critical), re-clamping the budgets`.
# Phase 2 (`MLX_SERVE_PREFIX_CACHE_PRESSURE=0`): repeats across warning and critical
#   overrides and asserts that no pressure budget step is logged.
# Exits 0 only on pass; last line is `PRESSURE-TEST-OK` or `PRESSURE-TEST-FAIL <why>`.
# shellcheck disable=SC1091

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_lib_models.sh"
source "$SCRIPT_DIR/lib/server.sh"

MODEL="${PRESSURE_TEST_MODEL:-${DSV41_TEST_MODEL:-${1:-}}}"
PORT="${PRESSURE_TEST_PORT:-${2:-19049}}"
BINARY="${MLX_SERVE_BINARY:-${BINARY:-./zig-out/bin/mlx-serve}}"
BASE="http://127.0.0.1:$PORT"
CACHE_ENTRIES="${PRESSURE_CACHE_ENTRIES:-4}"
CACHE_MB="${PRESSURE_CACHE_MB:-256}"
CTX_SIZE="${PRESSURE_CTX_SIZE:-32768}"
FILL_REQS="${PRESSURE_FILL_REQS:-5}"
PROMPT_WORDS="${PRESSURE_PROMPT_WORDS:-900}"
ON_WAIT_SECS="${PRESSURE_ON_WAIT_SECS:-25}"
OFF_HOLD_SECS="${PRESSURE_OFF_HOLD_SECS:-14}"
STEP_PATTERN='\[hot-cache\] memory pressure worse: (warning|critical), re-clamping the budgets'

SERVER_PID=""
LOGFILE=""
OUT=""
PRESSURE_FILE=""
FINAL_LINE="PRESSURE-TEST-FAIL unexpected exit"

EXTRA_FLAGS=()
[ -z "${PRESSURE_TEST_FLAGS:-}" ] || read -ra EXTRA_FLAGS <<< "$PRESSURE_TEST_FLAGS"

fail() {
    FINAL_LINE="PRESSURE-TEST-FAIL $*"
    exit 1
}

set_pressure() {
    local level="$1"
    local tmp="$PRESSURE_FILE.tmp"
    printf '%s\n' "$level" > "$tmp"
    mv -f "$tmp" "$PRESSURE_FILE"
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        srv_stop "$SERVER_PID"
        SERVER_PID=""
    fi
}

diagnostics() {
    local log
    for log in "$OUT"/*.server.log; do
        [ -f "$log" ] || continue
        echo "--- $log"
        tail -n 60 "$log" || true
    done
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM ERR
    [ "$FINAL_LINE" = "PRESSURE-TEST-OK" ] || [ "$rc" -ne 0 ] || rc=1
    stop_server
    if [ "$rc" -ne 0 ] && [ -n "$OUT" ] && [ -d "$OUT" ]; then
        diagnostics
        echo "artifacts: $OUT"
    elif [ -n "$OUT" ] && [ -d "$OUT" ]; then
        rm -rf "$OUT"
    fi
    echo "$FINAL_LINE"
    exit "$rc"
}

trap 'FINAL_LINE="PRESSURE-TEST-FAIL command failed at line $LINENO"' ERR
trap cleanup EXIT INT TERM

[ -x "$BINARY" ] || fail "binary not executable: $BINARY"

if [ -z "$MODEL" ]; then
    MODEL=$(find_fitting_model \
        mlx-community/gemma-4-e4b-it-4bit \
        mlx-community/gemma-4-e2b-it-4bit \
        lmstudio-community/Qwen3.5-2B-MLX-4bit \
        mlx-community/Qwen3-4B-4bit || true)
fi
[ -n "$MODEL" ] || fail "no model specified or found (set PRESSURE_TEST_MODEL or pass model dir)"
[ -f "$MODEL/config.json" ] || fail "missing config.json in $MODEL"

mkdir -p "$HOME/claude-tmp"
OUT=$(mktemp -d "$HOME/claude-tmp/prefix-cache-pressure.XXXXXX")
PRESSURE_FILE="$OUT/pressure.level"

check_port_free() {
    python3 - "$PORT" <<'PY' || fail "port $PORT is already in use"
import socket, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

start_server() {
    local label="$1"
    local case_home="$2"
    local pressure_env="$3"
    mkdir -p "$case_home"
    LOGFILE="$OUT/$label.server.log"
    set_pressure "normal"
    check_port_free
    if [ -n "$pressure_env" ]; then
        HOME="$case_home" \
            MLX_SERVE_PRESSURE_OVERRIDE="$PRESSURE_FILE" \
            MLX_SERVE_PREFIX_CACHE_PRESSURE="$pressure_env" \
            "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
            --ctx-size "$CTX_SIZE" --no-pld --no-mtp --no-drafter --no-vision --metrics \
            --prefix-cache-entries "$CACHE_ENTRIES" --prefix-cache-mem "${CACHE_MB}MB" \
            --prefix-cache-disk off --log-level info \
            ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} > "$LOGFILE" 2>&1 &
    else
        HOME="$case_home" \
            MLX_SERVE_PRESSURE_OVERRIDE="$PRESSURE_FILE" \
            "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
            --ctx-size "$CTX_SIZE" --no-pld --no-mtp --no-drafter --no-vision --metrics \
            --prefix-cache-entries "$CACHE_ENTRIES" --prefix-cache-mem "${CACHE_MB}MB" \
            --prefix-cache-disk off --log-level info \
            ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} > "$LOGFILE" 2>&1 &
    fi
    SERVER_PID=$!
    srv_wait_health "$BASE" 600 1 "$SERVER_PID" || fail "server failed health check in $label"
}

fill_hot_cache() {
    local phase="$1"
    python3 - "$BASE" "$FILL_REQS" "$PROMPT_WORDS" "$phase" <<'PY' || fail "chat request failed while filling cache ($phase)"
import json, sys, urllib.request

base, count, words, phase = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
for idx in range(1, count + 1):
    body_words = " ".join(f"item-{phase}-{idx}-{w}-{(idx * 31 + w * 17) % 997}" for w in range(words))
    prompt = f"Distinct session {phase}-{idx}:\n{body_words}\nReply with OK."
    payload = json.dumps({
        "model": "mlx-serve",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0.0,
        "enable_thinking": False,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    msg = data.get("choices", [{}])[0].get("message", {})
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    if not text:
        raise SystemExit(f"empty completion on request {idx}")
PY
    grep -q '\[hot-cache\] resident=' "$LOGFILE" || fail "hot cache showed no resident entries in $phase"
}

log_since_matches() {
    local mark="$1"
    local pattern="$2"
    tail -n "+$((mark + 1))" "$LOGFILE" > "$OUT/slice.log"
    grep -Eq "$pattern" "$OUT/slice.log"
}

wait_for_step_line() {
    local mark="$1"
    local tries=$(( ON_WAIT_SECS * 2 ))
    local _
    for _ in $(seq 1 "$tries"); do
        if log_since_matches "$mark" "$STEP_PATTERN"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

echo "== phase 1: default pressure gauge =="
start_server "phase1-on" "$OUT/on-home" ""
fill_hot_cache "on"
if grep -q '\[hot-cache\] memory pressure ' "$LOGFILE"; then
    fail "phase 1 (default): unexpected memory pressure step at normal"
fi
ON_MARK=$(wc -l < "$LOGFILE" | tr -d ' ')
set_pressure "warning"
if ! wait_for_step_line "$ON_MARK"; then
    fail "phase 1 (default): missing '$STEP_PATTERN' after ${ON_WAIT_SECS}s under warning override"
fi
MATCHED_STEP=$(grep -E "$STEP_PATTERN" "$OUT/slice.log" | tail -n 1)
echo "PASS phase 1: $MATCHED_STEP"
set_pressure "normal"
curl -sf -m 5 "$BASE/health" >/dev/null || fail "phase 1: server unhealthy after pressure override"
stop_server

echo "== phase 2: MLX_SERVE_PREFIX_CACHE_PRESSURE=0 =="
start_server "phase2-off" "$OUT/off-home" "0"
fill_hot_cache "off"
OFF_MARK=$(wc -l < "$LOGFILE" | tr -d ' ')
set_pressure "warning"
OFF_TICKS=$(( OFF_HOLD_SECS * 2 ))
for _ in $(seq 1 "$OFF_TICKS"); do
    if log_since_matches "$OFF_MARK" '\[hot-cache\] memory pressure '; then
        fail "phase 2 (MLX_SERVE_PREFIX_CACHE_PRESSURE=0): unexpected memory pressure step logged"
    fi
    sleep 0.5
done
set_pressure "critical"
for _ in $(seq 1 6); do
    if log_since_matches "$OFF_MARK" '\[hot-cache\] memory pressure '; then
        fail "phase 2 (MLX_SERVE_PREFIX_CACHE_PRESSURE=0): unexpected critical pressure step logged"
    fi
    sleep 0.5
done
set_pressure "normal"
curl -sf -m 5 "$BASE/health" >/dev/null || fail "phase 2: server unhealthy after pressure override"
if log_since_matches "$OFF_MARK" '\[hot-cache\] (memory pressure |budget revised )'; then
    fail "phase 2 (MLX_SERVE_PREFIX_CACHE_PRESSURE=0): unexpected budget revision logged"
fi
echo "PASS phase 2: MLX_SERVE_PREFIX_CACHE_PRESSURE=0 logged no budget step"
stop_server

FINAL_LINE="PRESSURE-TEST-OK"
