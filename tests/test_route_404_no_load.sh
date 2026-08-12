#!/bin/bash
# Guard: a path this server does not serve costs NOTHING.
#
# Model resolution ran ahead of dispatch, so a POST to a non-existent endpoint
# carrying a `model` field cold-loaded that model and only then answered 404.
# Found by typing /v1/load instead of /v1/load-model: 2m42s and 121 GB resident
# for a wrong URL, and a one-liner for any client to pin the box.
#
# The server boots headless over a --model-dir with NO --model, so every entry
# starts `unloaded` and the assertion is that a 404'd request leaves it that
# way. Nothing loads when the fix holds; the small default model loads when it
# doesn't, so the test is fast either way.
#
# Usage: ./tests/test_route_404_no_load.sh [model_dir] [port]

set -u

MODEL=${1:-~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
PORT=${2:-8132}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

MODEL=$(eval echo "$MODEL")
if [ ! -d "$MODEL" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi
if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq is required"; exit 1; }

ROOT=$(dirname "$(dirname "$MODEL")")
ID="$(basename "$(dirname "$MODEL")")/$(basename "$MODEL")"

./zig-out/bin/mlx-serve serve --port $PORT --host 127.0.0.1 --log-level info \
    --model-dir "$ROOT" >/tmp/mlx-serve-route404.log 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 60); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    if [ "$i" -eq 60 ]; then echo "FAIL: server did not start within 60s"; exit 1; fi
    sleep 1
done

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

state_of() { curl -s "$BASE/v1/models" | jq -r --arg id "$ID" '.data[] | select(.id==$id) | .state'; }

echo "=== a 404 endpoint must not load a model ==="
echo "target: $ID"

if [ "$(state_of)" != "unloaded" ]; then
    echo "FAIL: precondition — $ID is not 'unloaded' at boot (got '$(state_of)')"
    exit 1
fi

# The real route is /v1/load-model; every one of these is a plausible typo or
# probe, and each names a model the registry CAN resolve.
for p in /v1/load /v1/chat /v1/__probe__; do
    CODE=$(curl -s -o /tmp/route404-body.json -w '%{http_code}' -m 600 "$BASE$p" \
        -H "Content-Type: application/json" -d "{\"model\":\"$ID\"}")
    if [ "$CODE" = "404" ]; then
        run_test "POST $p -> 404" PASS ""
    else
        run_test "POST $p -> 404" FAIL "got HTTP $CODE: $(head -c 120 /tmp/route404-body.json)"
    fi
done

STATE=$(state_of)
if [ "$STATE" = "unloaded" ]; then
    run_test "no model was loaded by the 404s" PASS ""
else
    run_test "no model was loaded by the 404s" FAIL "$ID is now '$STATE'"
fi

# The routes that DO exist must be unaffected by the gate.
CODE=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/models")
[ "$CODE" = "200" ] && run_test "GET /v1/models still 200" PASS "" \
                    || run_test "GET /v1/models still 200" FAIL "got $CODE"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/props")
[ "$CODE" = "200" ] && run_test "GET /props still 200 with no default model" PASS "" \
                    || run_test "GET /props still 200 with no default model" FAIL "got $CODE"

echo ""
echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
