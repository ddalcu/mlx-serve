#!/bin/bash
# Plan 05 Phase F — concurrent requests across two loaded models.
#
# Two clients fire chat requests against different model ids in parallel.
# The scheduler's single inference thread must serialize them via the
# per-tick model fence; both must complete with non-empty content; no
# deadlock or stalled stream.
#
# Usage: ./tests/test_multi_model_concurrent.sh [model_dir_root] [port]

set -e

ROOT="${1:-$HOME/.mlx-serve/models}"
PORT="${2:-8097}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$ROOT" ]; then
    echo -e "${YELLOW}SKIP${NC} test_multi_model_concurrent: $ROOT not found."
    exit 0
fi

source "$(dirname "$0")/_lib_supported_models.sh"
MODELS=()
while IFS= read -r m; do MODELS+=("$m"); done < <(list_supported_models "$ROOT" 2)
if [ "${#MODELS[@]}" -lt 2 ]; then
    echo -e "${YELLOW}SKIP${NC} test_multi_model_concurrent: need >=2 models."
    exit 0
fi
M1="${MODELS[0]}"; M2="${MODELS[1]}"

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
LOGFILE=$(mktemp)
"$BINARY" --model-dir "$ROOT" --model "$ROOT/$M1" --serve --port "$PORT" \
    --max-resident-models 4 --max-concurrent 4 ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
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

# Warm both up sequentially first so cold-load doesn't race with the
# concurrent fire (the timing of two cold loads is well-tested elsewhere).
echo "  warming $M1 and $M2..."
curl -fs -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M1\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":4}" >/dev/null
curl -fs -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M2\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":4}" >/dev/null

OUT1=$(mktemp); OUT2=$(mktemp)
echo "  firing concurrent requests..."
curl -fs -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M1\",\"messages\":[{\"role\":\"user\",\"content\":\"Count 1 to 5.\"}],\"max_tokens\":40}" -o "$OUT1" &
PID1=$!
curl -fs -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M2\",\"messages\":[{\"role\":\"user\",\"content\":\"Count 1 to 5.\"}],\"max_tokens\":40}" -o "$OUT2" &
PID2=$!

wait $PID1 && S1=$? || S1=$?
wait $PID2 && S2=$? || S2=$?

FAIL=0
check() {
    local name="$1"; local status="$2"; local file="$3"
    if [ "$status" -ne 0 ]; then
        echo -e "${RED}FAIL${NC} $name: curl status $status"
        FAIL=1
        return
    fi
    local content
    content=$(python3 -c "import sys,json; print(json.load(open('$file'))['choices'][0]['message'].get('content','') or '')")
    if [ -z "$content" ]; then
        echo -e "${RED}FAIL${NC} $name: empty content"
        FAIL=1
        return
    fi
    echo -e "${GREEN}PASS${NC} $name: ${content:0:40}..."
}
check "concurrent M1" "$S1" "$OUT1"
check "concurrent M2" "$S2" "$OUT2"
rm -f "$OUT1" "$OUT2"

exit $FAIL
