#!/bin/bash
# Plan 05 Phase F — malformed model directory surfaces as a load failure.
#
# Creates a fake model directory containing a bogus config.json under a
# temp root, points --model-dir at it, then targets it via the chat API.
# The request must return 500 (model_load_failed) and the entry must
# appear in /v1/models with state="error". Subsequent requests against a
# valid sibling model still succeed (isolation).
#
# Usage: ./tests/test_multi_model_load_failure.sh [valid_model] [port]

set -e

VALID="${1:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}"
PORT="${2:-8098}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$VALID" ]; then
    echo -e "${YELLOW}SKIP${NC} test_multi_model_load_failure: $VALID not found."
    exit 0
fi

TMPROOT=$(mktemp -d)
VALID_ID=$(basename "$VALID")
ln -s "$VALID" "$TMPROOT/$VALID_ID"
mkdir -p "$TMPROOT/broken-model"
cat > "$TMPROOT/broken-model/config.json" <<EOF
{ "model_type": "not-a-real-model", "hidden_size": 1, "num_hidden_layers": 1 }
EOF

cleanup_root() { rm -rf "$TMPROOT"; }

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
LOGFILE=$(mktemp)
"$BINARY" --model-dir "$TMPROOT" --model "$TMPROOT/$VALID_ID" --serve --port "$PORT" \
    ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LOGFILE"
    cleanup_root
}
trap cleanup EXIT

for _ in $(seq 1 30); do
    curl -fs "$BASE/health" >/dev/null 2>&1 && break
    sleep 1
done

FAIL=0

echo "== request broken model returns 500 =="
HTTP_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"broken-model","messages":[{"role":"user","content":"Hi."}],"max_tokens":4}')
if [ "$HTTP_STATUS" = "500" ]; then
    echo -e "${GREEN}PASS${NC} broken-model → HTTP 500"
else
    echo -e "${RED}FAIL${NC} expected 500, got $HTTP_STATUS"
    FAIL=1
fi

echo
echo "== /v1/models reports broken-model state=error =="
MODELS_JSON=$(curl -fs "$BASE/v1/models")
STATE=$(echo "$MODELS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(next((e.get('state','?') for e in d if e['id']=='broken-model'), None))")
if [ "$STATE" = "error" ]; then
    echo -e "${GREEN}PASS${NC} broken-model: state=error"
else
    echo -e "${RED}FAIL${NC} broken-model state=$STATE (expected error)"
    FAIL=1
fi

echo
echo "== valid model still works =="
HTTP_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$VALID_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":4}")
if [ "$HTTP_STATUS" = "200" ]; then
    echo -e "${GREEN}PASS${NC} valid model isolated from broken one"
else
    echo -e "${RED}FAIL${NC} valid model status=$HTTP_STATUS"
    FAIL=1
fi

exit $FAIL
