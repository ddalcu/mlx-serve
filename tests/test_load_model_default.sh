#!/usr/bin/env bash
# `/v1/load-model` `"default": true` re-points the server's default model.
#
# The app's model picker hot-switches a running server instead of restarting
# it — but a hot-load alone leaves the DEFAULT untouched, so requests that
# omit `model` (or use the "mlx-serve" alias: the Claude Code launcher env,
# curl users) kept hitting the OLD model, and /v1/models kept sorting the old
# default first — which is exactly what the app's own pill/tray read back.
# The flag is explicit opt-in: a media-gen side-load loads BESIDE the chat
# model and must never steal the chat default.
#
# NEEDS REAL MODELS (only a load that SUCCEEDS may be promoted, so stub dirs
# can't reach the branch): skips cleanly when the two defaults are absent.
# Override with BOOT_MODEL / SWITCH_MODEL env vars.
#
# Usage: ./tests/test_load_model_default.sh [port]
set -uo pipefail
PORT="${1:-11381}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MODELS_ROOT="${MODELS_ROOT:-$HOME/.mlx-serve/models}"
BOOT_MODEL="${BOOT_MODEL:-$MODELS_ROOT/LiquidAI/LFM2.5-2.6B-MLX-mxfp4}"
SWITCH_MODEL="${SWITCH_MODEL:-$MODELS_ROOT/Jundot/Qwen3.6-27B-oQ4e-mtp}"
if [ ! -f "$BOOT_MODEL/config.json" ] || [ ! -f "$SWITCH_MODEL/config.json" ]; then
    echo "SKIP: needs two local chat models (BOOT_MODEL=$BOOT_MODEL, SWITCH_MODEL=$SWITCH_MODEL)"
    exit 0
fi

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

LOG="$(mktemp)"
SRV=""
cleanup() {
    [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    rm -f "$LOG"
}
trap cleanup EXIT
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 0.5

"$BIN" --serve --model "$BOOT_MODEL" --model-dir "$MODELS_ROOT" --port "$PORT" --log-file off >"$LOG" 2>&1 &
SRV=$!
UP=0
for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { UP=1; break; }
    kill -0 "$SRV" 2>/dev/null || break
    sleep 0.5
done
[ "$UP" = "1" ] || { echo "FAIL: server never became healthy"; tail -5 "$LOG"; exit 1; }

default_id() {
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c \
        "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])"
}
load() { # load <json body> — POST /v1/load-model, prints http code
    curl -s -o /dev/null -w '%{http_code}' --max-time 300 \
        -X POST "http://127.0.0.1:$PORT/v1/load-model" \
        -H 'Content-Type: application/json' -d "$1"
}

BOOT_ID="$(default_id)"
echo "boot default: $BOOT_ID"

# ── [1] A side-load WITHOUT the flag leaves the default alone.
CODE="$(load "{\"model\":\"$SWITCH_MODEL\"}")"
check "[1a] side-load returns 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
AFTER_SIDE="$(default_id)"
check "[1b] default unchanged after side-load ($AFTER_SIDE)" "$([ "$AFTER_SIDE" = "$BOOT_ID" ] && echo 1 || echo 0)"

# ── [2] `"default": true` promotes the loaded model to the default.
CODE="$(load "{\"model\":\"$SWITCH_MODEL\",\"default\":true}")"
check "[2a] default-load returns 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
AFTER_DEFAULT="$(default_id)"
check "[2b] default re-pointed ($AFTER_DEFAULT)" "$([ "$AFTER_DEFAULT" != "$BOOT_ID" ] && echo 1 || echo 0)"
SWITCH_BASE="$(basename "$SWITCH_MODEL")"
case "$AFTER_DEFAULT" in
    *"$SWITCH_BASE"*) check "[2c] new default is the switched model" 1 ;;
    *)                check "[2c] new default is the switched model (got $AFTER_DEFAULT)" 0 ;;
esac

# ── [3] A request that OMITS `model` routes to the new default. The response
# echoes the resolved model's config `model_type` when the request names none,
# so the arch tells the two models apart (lfm2 vs qwen3_5_moe).
ECHOED="$(curl -s --max-time 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}' | python3 -c \
    "import sys,json;print(json.load(sys.stdin).get('model',''))")"
check "[3] aliased request answered by the new default (model=$ECHOED)" \
    "$([ "$ECHOED" = "qwen3_5_moe" ] && echo 1 || echo 0)"

echo ""
echo "load-model default: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ] || exit 1
