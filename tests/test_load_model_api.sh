#!/bin/bash
# Plan 05 Phase F — POST /v1/load-model.
#
# Starts the server with a `--model-dir` containing >=2 models. Hits the
# explicit load endpoint for the second model and asserts the response
# shape + that a subsequent chat request to that id returns success
# (without the cold-load round-trip).
#
# Usage: ./tests/test_load_model_api.sh [model_dir_root] [port]

set -e

ROOT="${1:-$HOME/.mlx-serve/models}"
PORT="${2:-8101}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$ROOT" ]; then
    echo -e "${YELLOW}SKIP${NC} load-model-api: $ROOT not found."
    exit 0
fi
source "$(dirname "$0")/_lib_supported_models.sh"
MODELS=()
while IFS= read -r m; do MODELS+=("$m"); done < <(list_supported_models "$ROOT" 2)
if [ "${#MODELS[@]}" -lt 2 ]; then
    echo -e "${YELLOW}SKIP${NC} load-model-api: need >=2 models."
    exit 0
fi
M1="${MODELS[0]}"; M2="${MODELS[1]}"

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
LOGFILE=$(mktemp)
"$BINARY" --model-dir "$ROOT" --model "$ROOT/$M1" --serve --port "$PORT" \
    --max-resident-models 4 ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LOGFILE"
}
trap cleanup EXIT

for _ in $(seq 1 30); do
    curl -fs "$BASE/health" >/dev/null 2>&1 && break
    sleep 1
done

FAIL=0
echo "== POST /v1/load-model id=$M2 =="
RESP=$(curl -fs -X POST "$BASE/v1/load-model" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M2\"}")
if [ -z "$RESP" ]; then
    echo -e "${RED}FAIL${NC} empty response"
    FAIL=1
else
    LOADED=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['model']['loaded'])")
    STATE=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['model']['state'])")
    if [ "$LOADED" = "True" ] && [ "$STATE" = "ready" ]; then
        echo -e "${GREEN}PASS${NC} response: loaded=true state=ready"
    else
        echo -e "${RED}FAIL${NC} response: loaded=$LOADED state=$STATE; full: $RESP"
        FAIL=1
    fi
fi

echo
echo "== unknown id returns 404 =="
HTTP_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/load-model" \
    -H 'Content-Type: application/json' -d '{"model":"no-such-model-xyz"}')
if [ "$HTTP_STATUS" = "404" ]; then
    echo -e "${GREEN}PASS${NC} unknown id → 404"
else
    echo -e "${RED}FAIL${NC} unknown id → $HTTP_STATUS (expected 404)"
    FAIL=1
fi

echo
echo "== follow-up chat request hits the warm model =="
HTTP_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M2\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":4}")
if [ "$HTTP_STATUS" = "200" ]; then
    echo -e "${GREEN}PASS${NC} subsequent chat on $M2: 200"
else
    echo -e "${RED}FAIL${NC} chat on $M2: $HTTP_STATUS"
    FAIL=1
fi

exit $FAIL
