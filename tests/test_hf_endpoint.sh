#!/bin/bash
# Guard: HF_ENDPOINT moves every hub request, for `mlx-serve pull` and `POST /api/pull`.
#
# A local http.server stands in for a mirror (hf-mirror.com keeps the hub's
# `api/models/...` and `resolve/main/...` layout). The repo id exists nowhere
# else, so a binary that still asks huggingface.co fails the pull. HOME points
# at a scratch dir: nothing lands in ~/.mlx-serve/models.
#
# Usage: ./tests/test_hf_endpoint.sh [binary] [port]

set -u

BIN=${1:-./zig-out/bin/mlx-serve}
PORT=${2:-8137}
HUB_PORT=$((PORT + 1))
REPO=mlx-serve-test/hf-endpoint-fixture
PASS=0
FAIL=0
TOTAL=0

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi
command -v python3 >/dev/null 2>&1 || { echo "FAIL: python3 is required"; exit 1; }

WORK=$(mktemp -d)
HUB="$WORK/hub"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null && wait $SERVER_PID 2>/dev/null
    kill $HUB_PID 2>/dev/null; wait $HUB_PID 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT

mkdir -p "$HUB/api/models/$REPO/tree" "$HUB/$REPO/resolve/main"
printf '{"model_type":"llama"}' >"$HUB/$REPO/resolve/main/config.json"
head -c 4096 /dev/zero >"$HUB/$REPO/resolve/main/model.safetensors"
cat >"$HUB/api/models/$REPO/tree/main" <<EOF
[{"type":"file","path":"config.json","size":22},
 {"type":"file","path":"model.safetensors","size":4096,"lfs":{"size":4096}}]
EOF

python3 -I -m http.server "$HUB_PORT" --bind 127.0.0.1 --directory "$HUB" >"$WORK/hub.log" 2>&1 &
HUB_PID=$!
for i in $(seq 1 50); do
    curl -s -o /dev/null "http://127.0.0.1:$HUB_PORT/" && break
    if [ "$i" -eq 50 ]; then echo "FAIL: fake hub did not start"; exit 1; fi
    sleep 0.1
done

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

pulled() { [ -s "$1/.mlx-serve/models/$REPO/config.json" ] && [ -s "$1/.mlx-serve/models/$REPO/model.safetensors" ]; }

echo "=== HF_ENDPOINT moves every hub request ==="

echo "[1] mlx-serve pull (trailing slash on the endpoint)"
H1="$WORK/home1"; mkdir -p "$H1"
HOME="$H1" HF_ENDPOINT="http://127.0.0.1:$HUB_PORT/" "$BIN" pull "$REPO" >"$WORK/pull1.log" 2>&1
if pulled "$H1"; then run_test "pull fetched the listing and both files from the endpoint" PASS
else run_test "pull fetched the listing and both files from the endpoint" FAIL "$(tail -3 "$WORK/pull1.log" | tr '\n' ' ')"; fi
if grep -q "GET /api/models/$REPO/tree/main" "$WORK/hub.log" && grep -q "GET /$REPO/resolve/main/model.safetensors" "$WORK/hub.log"; then
    run_test "the endpoint served the listing and the weights" PASS
else run_test "the endpoint served the listing and the weights" FAIL "hub log: $(tr '\n' ' ' <"$WORK/hub.log" | cut -c1-300)"; fi

echo "[2] a value without an http(s) scheme is refused by name"
H2="$WORK/home2"; mkdir -p "$H2"
HOME="$H2" HF_ENDPOINT="127.0.0.1:$HUB_PORT" "$BIN" pull "$REPO" >"$WORK/pull2.log" 2>&1
if ! pulled "$H2" && grep -q "HF_ENDPOINT" "$WORK/pull2.log"; then run_test "bad HF_ENDPOINT named in the error, nothing pulled" PASS
else run_test "bad HF_ENDPOINT named in the error, nothing pulled" FAIL "$(tail -3 "$WORK/pull2.log" | tr '\n' ' ')"; fi

echo "[3] POST /api/pull on a server started with HF_ENDPOINT"
H3="$WORK/home3"; mkdir -p "$H3/models"
HOME="$H3" HF_ENDPOINT="http://127.0.0.1:$HUB_PORT" "$BIN" serve --port "$PORT" --host 127.0.0.1 \
    --model-dir "$H3/models" >"$WORK/serve.log" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    if [ "$i" -eq 60 ]; then echo "FAIL: server did not start within 60s"; tail -5 "$WORK/serve.log"; exit 1; fi
    sleep 1
done
BODY=$(curl -s -X POST "http://127.0.0.1:$PORT/api/pull" -d "{\"model\":\"$REPO\",\"stream\":false}")
if pulled "$H3"; then run_test "/api/pull fetched from the endpoint" PASS
else run_test "/api/pull fetched from the endpoint" FAIL "response: $BODY"; fi

echo ""
echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
