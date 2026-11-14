#!/bin/bash
# Guard: HF_ENDPOINT moves every hub request, for `mlx-serve pull` and `POST /api/pull`.
#
# A local http.server stands in for a mirror (hf-mirror.com keeps the hub's
# `api/models/...` and `resolve/main/...` layout). The repo id exists nowhere
# else, so a binary that still asks huggingface.co fails the pull. HOME points
# at a scratch dir: nothing lands in ~/.mlx-serve/models.
#
# Usage: ./tests/test_hf_endpoint.sh [port]

set -u
source "$(dirname "$0")/lib/server.sh"

BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
PORT=${1:-8137}
HUB_PORT=$((PORT + 1))
REPO=mlx-serve-test/hf-endpoint-fixture
PASS=0
FAIL=0

[ -x "$BIN" ] || { echo "FAIL: $BIN not built — run 'zig build -Doptimize=ReleaseFast' first"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "FAIL: python3 is required"; exit 1; }

WORK=$(mktemp -d)
HUB="$WORK/hub"
SERVER_PID=""
HUB_PID=""
cleanup() {
    srv_stop "$SERVER_PID"
    srv_stop "$HUB_PID"
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
    [ "$i" -eq 50 ] && { echo "FAIL: fake hub did not start"; exit 1; }
    sleep 0.1
done

check() {
    if [ "$2" = 1 ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}
pulled() { [ -s "$1/.mlx-serve/models/$REPO/config.json" ] && [ -s "$1/.mlx-serve/models/$REPO/model.safetensors" ]; }

echo "[1] mlx-serve pull (trailing slash on the endpoint)"
H1="$WORK/home1"; mkdir -p "$H1"
HOME="$H1" HF_ENDPOINT="http://127.0.0.1:$HUB_PORT/" "$BIN" pull "$REPO" >"$WORK/pull1.log" 2>&1
pulled "$H1"; check "pull fetched the listing and both files from the endpoint" $((1 - $?)) "$(tail -3 "$WORK/pull1.log" | tr '\n' ' ')"
grep -q "GET /api/models/$REPO/tree/main" "$WORK/hub.log" && grep -q "GET /$REPO/resolve/main/model.safetensors" "$WORK/hub.log"
check "the endpoint served the listing and the weights" $((1 - $?)) "hub log: $(tr '\n' ' ' <"$WORK/hub.log" | cut -c1-300)"

echo "[2] a value without an http(s) scheme is refused by name"
H2="$WORK/home2"; mkdir -p "$H2"
HOME="$H2" HF_ENDPOINT="127.0.0.1:$HUB_PORT" "$BIN" pull "$REPO" >"$WORK/pull2.log" 2>&1
! pulled "$H2" && grep -q "HF_ENDPOINT" "$WORK/pull2.log"
check "bad HF_ENDPOINT named in the error, nothing pulled" $((1 - $?)) "$(tail -3 "$WORK/pull2.log" | tr '\n' ' ')"

echo "[3] POST /api/pull on a server started with HF_ENDPOINT"
H3="$WORK/home3"; mkdir -p "$H3/models"
HOME="$H3" HF_ENDPOINT="http://127.0.0.1:$HUB_PORT" "$BIN" serve --port "$PORT" --host 127.0.0.1 \
    --model-dir "$H3/models" --log-file off >"$WORK/serve.log" 2>&1 &
SERVER_PID=$!
srv_wait_health "http://127.0.0.1:$PORT" 60 0.5 "$SERVER_PID" || { echo "FAIL: server did not start"; tail -5 "$WORK/serve.log"; exit 1; }
BODY=$(curl -s -X POST "http://127.0.0.1:$PORT/api/pull" -d "{\"model\":\"$REPO\",\"stream\":false}")
pulled "$H3"; check "/api/pull fetched from the endpoint" $((1 - $?)) "response: $BODY"

echo "=== $PASS/$((PASS + FAIL)) passed ==="
[ "$FAIL" -eq 0 ]
