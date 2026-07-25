#!/bin/bash
# Guard: the built-in console at `GET /` must render on a server with NO model.
#
# `GET /` was dispatched AFTER model resolution and rendered one *LoadedModel,
# so a headless boot — `mlx-serve serve` / `--serve --model-dir`, now the
# default way the server starts and the only way the app launches it — answered
# 503 {"error":"No default model configured"} at the root. The page is the
# thing a person opens first, and it is also the model PICKER, so it has to
# render before anything is loaded, by construction.
#
# Also pins the two properties a page rewrite can silently drop:
#   * every endpoint the server serves is documented (the reference had been
#     missing the whole Ollama /api/* surface);
#   * the live-metrics mount is present with --metrics and absent without it
#     (deliberately duplicates one test_metrics.sh assertion — that script
#     needs a real checkpoint, this one doesn't).
#
# FULLY HERMETIC: an empty --model-dir discovers zero models, so no weights are
# needed and the whole thing runs in seconds (same trick as
# tests/test_headless_spec_flags.sh).
#
# Usage: ./tests/test_index_page.sh [port]

set -u

PORT="${1:-11266}"
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

EMPTY_DIR="$(mktemp -d)"
LOG="$(mktemp)"
BODY="$(mktemp)"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    rm -rf "$EMPTY_DIR" "$LOG" "$BODY"
}
trap cleanup EXIT

boot() {
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    sleep 0.5
    : > "$LOG"
    "$BINARY" --serve --model-dir "$EMPTY_DIR" --port "$PORT" --log-file off "$@" > "$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 0.5
        kill -0 "$SERVER_PID" 2>/dev/null || break
    done
    echo "  (server did not come up; log follows)"; cat "$LOG"
    return 1
}

stop() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}

echo "Built-in console at GET / (port $PORT, no model)"

# ── 1. It renders at all without a model ────────────────────────────────────
echo "[1/3] headless GET /"
if boot; then
    STATUS=$(curl -s -o "$BODY" -w '%{http_code}' "http://127.0.0.1:$PORT/")
    CT=$(curl -s -D - -o /dev/null "http://127.0.0.1:$PORT/" | grep -i '^content-type:' | tr -d '\r')
    check "GET / with no model loaded → 200 (got $STATUS)" \
        "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"
    check "Content-Type is text/html" \
        "$(echo "$CT" | grep -qi 'text/html' && echo 1 || echo 0)"
    check "no 'No default model configured' in the body" \
        "$(grep -q 'No default model configured' "$BODY" && echo 0 || echo 1)"

    # ── 2. The console + the full endpoint reference are in the page ────────
    echo "[2/3] console markup + endpoint coverage"
    for tab in chat monitor api; do
        grep -q "data-tab=\"$tab\"" "$BODY"
        check "sidebar destination '$tab' present" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    done
    # Chat opens in its empty state before any JS runs.
    grep -q '<section class="panel active" id=tab-chat>' "$BODY"
    check "chat is the default panel" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    grep -Eq 'id="?chat-empty"?' "$BODY"
    check "chat has a simple empty state" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    grep -Eq 'id="?recent-list"?' "$BODY"
    check "sidebar has a Recents list" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    # Media work is natural language in the chat, not its own destination.
    grep -q 'data-tab="images"' "$BODY"
    check "no separate images tab" "$([ $? -ne 0 ] && echo 1 || echo 0)"
    grep -q 'data-tab="audio"' "$BODY"
    check "no separate audio tab" "$([ $? -ne 0 ] && echo 1 || echo 0)"
    # Quote-tolerant: the page mixes quoted and bare attribute values.
    grep -Eq 'id="?chat-send"?' "$BODY"
    check "chat composer present" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    grep -Eq 'id="?chat-model"?' "$BODY"
    check "model picker present" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    grep -Eq 'id="?chat-files"?' "$BODY"
    check "image attach control present" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    grep -Eq 'id="?mon-models"?' "$BODY"
    check "monitor model table present" "$([ $? -eq 0 ] && echo 1 || echo 0)"
    # Ours now — a user-facing system prompt box would fight it. Sampling knobs
    # went with it: this is a console, not a tuning rig.
    grep -Eq 'id="?chat-system"?' "$BODY"
    check "no user system-prompt box" "$([ $? -ne 0 ] && echo 1 || echo 0)"
    grep -Eq 'id="?chat-temp"?|id="?chat-maxtok"?' "$BODY"
    check "no temperature / max-tokens inputs" "$([ $? -ne 0 ] && echo 1 || echo 0)"
    grep -q '/v1/models' "$BODY"
    check "console fetches the model list" "$([ $? -eq 0 ] && echo 1 || echo 0)"

    # The Ollama surface is the one the hand-written reference had omitted
    # wholesale. Zig's `index page documents every endpoint` test pins this
    # against ROUTE_PATHS; this is the served-bytes end of the same claim.
    MISSING=""
    for ep in /api/chat /api/generate /api/tags /api/show /api/ps /api/pull \
              /api/version /api/embed /api/embeddings; do
        grep -q "$ep" "$BODY" || MISSING="$MISSING $ep"
    done
    check "every /api/* endpoint documented (missing:${MISSING:-none})" \
        "$([ -z "$MISSING" ] && echo 1 || echo 0)"

    MISSING=""
    for ep in /v1/chat/completions /v1/completions /v1/responses /v1/messages \
              /v1/embeddings /v1/images/generations /v1/images/edits \
              /v1/audio/speech /v1/audio/music-generations /v1/video/generations \
              /v1/3d/generations /v1/load-model /v1/unload-model /tokenize \
              /detokenize /props /health /metrics.json; do
        grep -q "$ep" "$BODY" || MISSING="$MISSING $ep"
    done
    check "every non-Ollama endpoint documented (missing:${MISSING:-none})" \
        "$([ -z "$MISSING" ] && echo 1 || echo 0)"

    check "no metrics panel mount without --metrics" \
        "$(grep -q 'id=mlx-metrics' "$BODY" && echo 0 || echo 1)"
    stop
else
    check "headless boot" 0
fi

# ── 3. --metrics puts the live panel in the header ──────────────────────────
echo "[3/3] --metrics panel mount"
if boot --metrics; then
    STATUS=$(curl -s -o "$BODY" -w '%{http_code}' "http://127.0.0.1:$PORT/")
    check "page still 200 with --metrics (got $STATUS)" \
        "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"
    check "metrics mount present with --metrics" \
        "$(grep -q 'id=mlx-metrics' "$BODY" && echo 1 || echo 0)"
    check "panel markup injected (m-status tile)" \
        "$(grep -q 'm-status' "$BODY" && echo 1 || echo 0)"
    stop
else
    check "boot with --metrics" 0
fi

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
