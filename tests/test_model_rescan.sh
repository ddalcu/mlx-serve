#!/bin/bash
# Guard: POST /v1/models/rescan absorbs models downloaded AFTER boot.
#
# Discovery walks the --model-dir roots ONCE at startup, so a model the app's
# Model Browser pulls while the server runs was invisible to /v1/models (and
# to the app's media pickers, which read it) until a restart. The rescan
# endpoint re-walks the roots and registers new dirs as unloaded stubs —
# add-only, idempotent, and it must never disturb existing entries.
#
# FULLY HERMETIC: an empty --model-dir boots headless in seconds; the "new
# model" is a fake dir (config.json + a stub safetensors) — stubs never load
# weights, so nothing real is needed.
#
# Usage: ./tests/test_model_rescan.sh [port]

set -u

PORT="${1:-11267}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

ROOT_DIR="$(mktemp -d)"
LOG="$(mktemp)"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    rm -rf "$ROOT_DIR" "$LOG"
}
trap cleanup EXIT

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 0.5
"$BINARY" --serve --model-dir "$ROOT_DIR" --port "$PORT" --log-file off > "$LOG" 2>&1 &
SERVER_PID=$!
UP=0
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { UP=1; break; }
    sleep 0.5
    kill -0 "$SERVER_PID" 2>/dev/null || break
done
if [ "$UP" != "1" ]; then
    echo "[fail] server did not come up; log follows"; cat "$LOG"; exit 1
fi

echo "POST /v1/models/rescan (port $PORT)"

# Boot saw an empty root.
MODELS=$(curl -s "http://127.0.0.1:$PORT/v1/models")
check "boot: /v1/models has no entries" \
    "$(echo "$MODELS" | grep -q '"id"' && echo 0 || echo 1)"

# A model lands on disk after boot (what a Model Browser download does).
mkdir -p "$ROOT_DIR/org/late-model"
echo '{"model_type":"minimax_h3"}' > "$ROOT_DIR/org/late-model/config.json"
echo "stub" > "$ROOT_DIR/org/late-model/transformer.safetensors"

RESCAN=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/models/rescan")
check "rescan reports the new model ({\"added\":1}, got $RESCAN)" \
    "$(echo "$RESCAN" | grep -q '"added":1' && echo 1 || echo 0)"

MODELS=$(curl -s "http://127.0.0.1:$PORT/v1/models")
check "/v1/models now lists org/late-model" \
    "$(echo "$MODELS" | grep -q '"org/late-model"' && echo 1 || echo 0)"
check "the stub advertises its media capability" \
    "$(echo "$MODELS" | grep -q '"video"' && echo 1 || echo 0)"

# Idempotent: nothing new → nothing added.
RESCAN=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/models/rescan")
check "second rescan adds nothing (got $RESCAN)" \
    "$(echo "$RESCAN" | grep -q '"added":0' && echo 1 || echo 0)"

# An INCOMPLETE media pack (media model_type, required marker missing — an
# in-flight download or a turbo-lora fragment) stays invisible to rescan, and
# a load by absolute path is refused by name instead of falling through to
# the text loader (which globs whatever safetensors exist and DIES on the
# first missing weight — live 2026-08-08).
mkdir -p "$ROOT_DIR/org/fragment"
echo '{"model_type":"minimax_h3"}' > "$ROOT_DIR/org/fragment/config.json"
echo "stub" > "$ROOT_DIR/org/fragment/turbo_lora.safetensors"

RESCAN=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/models/rescan")
check "incomplete media pack is not absorbed (got $RESCAN)" \
    "$(echo "$RESCAN" | grep -q '"added":0' && echo 1 || echo 0)"

LOAD=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/load-model" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ROOT_DIR/org/fragment\"}")
check "load-by-path refuses the incomplete pack by name (got $LOAD)" \
    "$(echo "$LOAD" | grep -qi 'incomplete media pack' && echo 1 || echo 0)"
HEALTH=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")
check "server survived the refused load (health $HEALTH)" \
    "$([ "$HEALTH" = "200" ] && echo 1 || echo 0)"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ] || exit 1
