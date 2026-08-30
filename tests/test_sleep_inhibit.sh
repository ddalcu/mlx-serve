#!/bin/bash
# The Mac stays awake only while mlx-serve has work in flight.
# Usage: ./tests/test_sleep_inhibit.sh [model_dir] [port]

set -u

MODEL=${1:-"$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"}
PORT=${2:-8138}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi
if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi
command -v pmset >/dev/null 2>&1 || { echo "SKIP: pmset not available (macOS only)"; exit 0; }

ROOT=$(dirname "$(dirname "$MODEL")")
ID="$(basename "$(dirname "$MODEL")")/$(basename "$MODEL")"
# The type also matches powerd, so check the unique name.
NAME="mlx-serve is generating"

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

held() { pmset -g assertions 2>/dev/null | grep -q "$NAME" && echo 1 || echo 0; }

start_server() {
    ./zig-out/bin/mlx-serve serve --port "$PORT" --host 127.0.0.1 --log-level info \
        --model-dir "$ROOT" "$@" >/tmp/mlx-serve-sleep-inhibit.log 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 120); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    echo "FAIL: server did not start within 60s"; exit 1
}

stop_server() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap stop_server EXIT

generate() {
    curl -s -N -X POST "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long story about a lighthouse keeper.\"}],\"max_tokens\":800,\"stream\":true}" \
        -o /tmp/mlx-serve-sleep-inhibit-stream.json
}

echo "=== idle-sleep inhibition ($ID, port $PORT) ==="

start_server
[ "$(held)" = "0" ] && run_test "idle server holds no assertion" PASS "" \
                    || run_test "idle server holds no assertion" FAIL "assertion held with nothing in flight"

generate &
REQ=$!
SEEN=0
for _ in $(seq 1 400); do
    [ "$(held)" = "1" ] && SEEN=1
    kill -0 $REQ 2>/dev/null || break
    sleep 0.1
done
wait $REQ
[ "$SEEN" = "1" ] && run_test "assertion held during generation" PASS "" \
                  || run_test "assertion held during generation" FAIL "never held while a request was in flight"

sleep 2
[ "$(held)" = "0" ] && run_test "assertion released once the server goes idle" PASS "" \
                    || run_test "assertion released once the server goes idle" FAIL "still held 2s after the last token"
stop_server

# Log order avoids racing the short startup load.
start_server --model "$MODEL" --log-level debug
ARM=$(grep -n -m1 'idle-sleep assertion held' /tmp/mlx-serve-sleep-inhibit.log | cut -d: -f1)
READY=$(grep -n -m1 'Model ready' /tmp/mlx-serve-sleep-inhibit.log | cut -d: -f1)
if [ -n "$ARM" ] && [ -n "$READY" ] && [ "$ARM" -lt "$READY" ]; then
    run_test "startup model load is covered" PASS ""
else
    run_test "startup model load is covered" FAIL "arm=${ARM:-none} ready=${READY:-none}"
fi
stop_server

echo "=== --no-prevent-sleep ==="
start_server --no-prevent-sleep
generate &
REQ=$!
SEEN=0
for _ in $(seq 1 400); do
    [ "$(held)" = "1" ] && SEEN=1
    kill -0 $REQ 2>/dev/null || break
    sleep 0.1
done
wait $REQ
[ "$SEEN" = "0" ] && run_test "opt-out never holds the assertion" PASS "" \
                  || run_test "opt-out never holds the assertion" FAIL "assertion held with --no-prevent-sleep"
stop_server

echo
echo "sleep-inhibit: $PASS/$TOTAL passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
