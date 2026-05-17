#!/bin/bash
# Plan 05 Phase F — multi-model routing.
#
# Starts the server with `--model-dir <dir>` containing >=2 discoverable
# models. Sends one chat-completion request targeting each model id, and
# asserts:
#   * Both requests return non-empty content (model loaded successfully).
#   * `/v1/models` shows BOTH as `loaded=true` after the requests.
#
# Usage:
#   ./tests/test_multi_model_basic.sh [model_dir_root] [port]
# Defaults: root = ~/.mlx-serve/models, port = 8095.

set -e

ROOT="${1:-$HOME/.mlx-serve/models}"
PORT="${2:-8095}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$ROOT" ]; then
    echo -e "${YELLOW}SKIP${NC} test_multi_model_basic: $ROOT not found."
    exit 0
fi

# Find the first two model dirs with a supported model_type — see
# tests/_lib_supported_models.sh and src/model_discovery.zig. Picking by
# config.json existence alone is not enough: a partial download of an
# unsupported arch (e.g. deepseek_v4) would crash mlx-serve on load.
source "$(dirname "$0")/_lib_supported_models.sh"
MODELS=()
while IFS= read -r m; do MODELS+=("$m"); done < <(list_supported_models "$ROOT" 2)
if [ "${#MODELS[@]}" -lt 2 ]; then
    echo -e "${YELLOW}SKIP${NC} test_multi_model_basic: need >=2 models under $ROOT (found ${#MODELS[@]})."
    exit 0
fi

M1="${MODELS[0]}"
M2="${MODELS[1]}"
echo "  using models: $M1, $M2"

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first."
    exit 1
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
LOGFILE=$(mktemp)
"$BINARY" --model-dir "$ROOT" --model "$ROOT/$M1" --serve --port "$PORT" --max-resident-models 4 ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LOGFILE"
}
trap cleanup EXIT

# Wait for server.
for _ in $(seq 1 30); do
    if curl -fs "$BASE/health" >/dev/null 2>&1; then break; fi
    sleep 1
done
if ! curl -fs "$BASE/health" >/dev/null 2>&1; then
    echo -e "${RED}FAIL${NC} server did not start in 30s. Log:"
    tail -30 "$LOGFILE"
    exit 1
fi

fire() {
    local model="$1"
    echo -n "  request to model=$model ... "
    local resp
    resp=$(curl -fs -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say one word.\"}],\"max_tokens\":8,\"temperature\":0.0}" 2>&1)
    if [ -z "$resp" ]; then
        echo -e "${RED}FAIL${NC} empty response"
        return 1
    fi
    local content
    content=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"].get("content","") or "")')
    if [ -z "$content" ]; then
        echo -e "${RED}FAIL${NC} empty content. Response: $resp"
        return 1
    fi
    echo -e "${GREEN}PASS${NC} (content=${content:0:30}...)"
}

FAIL=0
echo "== request each model =="
fire "$M1" || FAIL=1
fire "$M2" || FAIL=1

echo
echo "== /v1/models reports both loaded =="
MODELS_JSON=$(curl -fs "$BASE/v1/models")
for m in "$M1" "$M2"; do
    LOADED=$(echo "$MODELS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(next((e['loaded'] for e in d if e['id']=='$m'), None))")
    if [ "$LOADED" = "True" ]; then
        echo -e "${GREEN}PASS${NC} $m: loaded=true"
    else
        echo -e "${RED}FAIL${NC} $m: loaded=$LOADED"
        FAIL=1
    fi
done

exit $FAIL
