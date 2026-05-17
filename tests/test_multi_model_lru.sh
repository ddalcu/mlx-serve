#!/bin/bash
# Plan 05 Phase F — LRU eviction.
#
# Server: --max-resident-models 2 with >=3 models in --model-dir. Hit each
# in sequence; assert the third request evicts the first (the server log
# shows the eviction line and /v1/models reports the first as loaded=false).
#
# Usage: ./tests/test_multi_model_lru.sh [model_dir_root] [port]

set -e

ROOT="${1:-$HOME/.mlx-serve/models}"
PORT="${2:-8096}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$ROOT" ]; then
    echo -e "${YELLOW}SKIP${NC} test_multi_model_lru: $ROOT not found."
    exit 0
fi

source "$(dirname "$0")/_lib_supported_models.sh"
MODELS=()
while IFS= read -r m; do MODELS+=("$m"); done < <(list_supported_models "$ROOT" 3)
if [ "${#MODELS[@]}" -lt 3 ]; then
    echo -e "${YELLOW}SKIP${NC} test_multi_model_lru: need >=3 models (found ${#MODELS[@]})."
    exit 0
fi

M1="${MODELS[0]}"; M2="${MODELS[1]}"; M3="${MODELS[2]}"
echo "  using models: $M1, $M2, $M3 (cap=2)"

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
LOGFILE=$(mktemp)
"$BINARY" --model-dir "$ROOT" --model "$ROOT/$M1" --serve --port "$PORT" \
    --max-resident-models 2 ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LOGFILE"
}
trap cleanup EXIT

for _ in $(seq 1 30); do
    if curl -fs "$BASE/health" >/dev/null 2>&1; then break; fi
    sleep 1
done

fire() {
    curl -fs -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":4,\"temperature\":0.0}" >/dev/null
}

FAIL=0
echo "== sequence: M1, M2, M3 (forces M1 eviction) =="
echo -n "  hit M1 (already resident from --model)... "; fire "$M1" && echo -e "${GREEN}OK${NC}" || { echo -e "${RED}FAIL${NC}"; FAIL=1; }
echo -n "  hit M2 (cold-load, fills cap)        ... "; fire "$M2" && echo -e "${GREEN}OK${NC}" || { echo -e "${RED}FAIL${NC}"; FAIL=1; }
echo -n "  hit M3 (cold-load, evicts M1)        ... "; fire "$M3" && echo -e "${GREEN}OK${NC}" || { echo -e "${RED}FAIL${NC}"; FAIL=1; }

echo
echo "== eviction log line =="
if grep -q "evicting model id=$M1" "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} server log shows M1 eviction"
else
    echo -e "${RED}FAIL${NC} expected eviction log line for $M1"
    grep -i "evict\|resident" "$LOGFILE" | head -5
    FAIL=1
fi

echo
echo "== /v1/models reflects state =="
MODELS_JSON=$(curl -fs "$BASE/v1/models")
M1_LOADED=$(echo "$MODELS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(next((e['loaded'] for e in d if e['id']=='$M1'), None))")
M3_LOADED=$(echo "$MODELS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(next((e['loaded'] for e in d if e['id']=='$M3'), None))")
[ "$M1_LOADED" = "False" ] && echo -e "${GREEN}PASS${NC} M1 loaded=false post-eviction" || { echo -e "${RED}FAIL${NC} M1 loaded=$M1_LOADED (expected False)"; FAIL=1; }
[ "$M3_LOADED" = "True" ]  && echo -e "${GREEN}PASS${NC} M3 loaded=true" || { echo -e "${RED}FAIL${NC} M3 loaded=$M3_LOADED"; FAIL=1; }

echo
echo "== reload M1 (evicts M2) =="
fire "$M1" && echo -e "${GREEN}OK${NC} reload-M1 returned non-empty" || { echo -e "${RED}FAIL${NC} reload-M1 failed"; FAIL=1; }
if grep -q "evicting model id=$M2" "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} M2 evicted on reload"
else
    echo -e "${RED}FAIL${NC} expected M2 eviction log line"
    FAIL=1
fi

exit $FAIL
