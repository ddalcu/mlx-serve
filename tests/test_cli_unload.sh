#!/bin/bash
# `mlx-serve unload <model>` — client of an already-running server.
#
# [1] nothing listening → exit 1, names the missing server. No weights.
# [2] live `--model` server: unload the resident id from GET /v1/models,
#     print `unloaded <id>`, /health still answers, the row is state=unloaded.
# [3] a second unload of that id exits 0 (already unloaded is success).
#
# The live arm skips when the small chat model is absent. [1] always runs.

set -u

MODEL_DIR=${1:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
PORT=${2:-8098}
BASE="http://127.0.0.1:$PORT"
BIN=./zig-out/bin/mlx-serve
PASS=0
FAIL=0

if [ ! -x "$BIN" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

run_test() {
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

# ── [1] closed port ──
DEAD="http://127.0.0.1:59999"
OUT=$("$BIN" unload some-id --url "$DEAD" 2>&1)
if [ $? -ne 0 ] && echo "$OUT" | grep -F -q "no mlx-serve server at $DEAD"; then
    run_test "closed port names the missing server" PASS
else
    run_test "closed port names the missing server" FAIL "$OUT"
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: model not found at $MODEL_DIR (live unload)"
    echo "== $PASS passed, $FAIL failed =="
    [ "$FAIL" = 0 ]
    exit $?
fi

echo "Starting server..."
"$BIN" --model "$MODEL_DIR" --serve --host 127.0.0.1 --port "$PORT" >/tmp/mlx-serve-unload-test.log 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 120); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    if [ "$i" -eq 120 ]; then
        echo "FAIL: server did not start"
        tail -20 /tmp/mlx-serve-unload-test.log
        exit 1
    fi
    sleep 1
done

ID=""
for i in $(seq 1 30); do
    ID=$(curl -sf "$BASE/v1/models" | python3 -c '
import json, sys
raw = sys.stdin.read()
if not raw.strip():
    sys.exit(0)
data = json.loads(raw).get("data") or []
ready = [m["id"] for m in data if m.get("state") == "ready" or m.get("loaded") is True]
if len(ready) == 1:
    print(ready[0])
')
    [ -n "$ID" ] && break
    sleep 1
done
if [ -z "$ID" ]; then
    echo "FAIL: no single ready model in /v1/models"
    curl -s "$BASE/v1/models"
    echo
    exit 1
fi

# ── [2] unload the resident model ──
OUT=$("$BIN" unload "$ID" --url "$BASE" 2>&1)
EC=$?
STATE=$(curl -sf "$BASE/v1/models" | ID="$ID" python3 -c '
import json, os, sys
want = os.environ["ID"]
data = json.loads(sys.stdin.read()).get("data") or []
for m in data:
    if m.get("id") == want:
        print(m.get("state", ""))
        break
else:
    print("MISSING")
')
HEALTH=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/health" || true)
if [ "$EC" -eq 0 ] && echo "$OUT" | grep -F -q "unloaded $ID" && [ "$STATE" = "unloaded" ] && [ "$HEALTH" = "200" ]; then
    run_test "unload frees the resident model and leaves the server up" PASS
else
    run_test "unload frees the resident model and leaves the server up" FAIL "ec=$EC state=$STATE health=$HEALTH out=$OUT"
fi

# ── [3] already unloaded ──
OUT=$("$BIN" unload "$ID" --url "$BASE" 2>&1)
if [ $? -eq 0 ] && echo "$OUT" | grep -F -q "unloaded $ID" && curl -sf "$BASE/health" >/dev/null; then
    run_test "second unload of the same id exits 0" PASS
else
    run_test "second unload of the same id exits 0" FAIL "$OUT"
fi

echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" = 0 ]
