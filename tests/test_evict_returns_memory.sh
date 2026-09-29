#!/bin/bash
# Make-room eviction must return the victim's memory to the OS before the next
# load's preflight reads OS-level availability. Two red bars, either one fails
# the run: (1) the load after a make-room eviction must answer 200 — without
# the clear the preflight reads OS availability with the victim's bytes still
# parked in the allocator pool and refuses a load that fits (the live
# 503-run of InsufficientMemory refusals); (2) window samples of
# mlx_serve:mlx_cache_bytes across the evict→ready window must stay BELOW the
# pool bar — a parked pool is the defect even when the next load happens to
# fit. The sampler's own /metrics read contends on the registry mutex held
# across the window, so samples are sparse, not absent; a pool-sized parking
# holds long enough to be caught.
#
# Usage: ./tests/test_evict_returns_memory.sh [port]
# Env: EVICT_TEST_ROOT (model root), EVICT_VICTIM=<org/repo>, EVICT_NEXT=<org/repo>,
#      EVICT_CACHE_LIMIT_MB (default 1024)
set -u

PORT="${1:-11293}"
BASE="http://127.0.0.1:$PORT"
ROOT="${EVICT_TEST_ROOT:-$HOME/.mlx-serve/models}"
VICTIM="${EVICT_VICTIM:-ddalcu/Qwen3.8-27B-MLX-Serve-4bit}"
NEXT="${EVICT_NEXT:-prism-ml/Ternary-Bonsai-2-27B-mlx-2bit}"
CACHE_LIMIT_MB="${EVICT_CACHE_LIMIT_MB:-1024}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

source "$(dirname "$0")/_lib_models.sh"
VICTIM_DIR=$(find_model "$VICTIM")
NEXT_DIR=$(find_model "$NEXT")
[ -d "$VICTIM_DIR" ] || { echo -e "${YELLOW}SKIP${NC}: victim $VICTIM not on any root"; exit 0; }
[ -d "$NEXT_DIR" ] || { echo -e "${YELLOW}SKIP${NC}: next model $NEXT not on any root"; exit 0; }
model_disk_gb() { du -sk "$1" 2>/dev/null | awk '{printf "%d", $1/1048576}'; }
V_GB=$(model_disk_gb "$VICTIM_DIR")
N_GB=$(model_disk_gb "$NEXT_DIR")
# Two loads must fit beside each other in the plan and on the box.
BUDGET_GB=$(sysctl -n hw.memsize | awk '{printf "%d", $1/1073741824*3/4}')
[ $((V_GB + N_GB)) -lt "$BUDGET_GB" ] || {
    echo -e "${YELLOW}SKIP${NC}: victim ${V_GB}G + next ${N_GB}G exceed the ${BUDGET_GB}G budget"; exit 0; }

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
LOG=$(mktemp)
SAMPLER=$(mktemp)
"$BINARY" --model-dir "$ROOT" --serve --host 127.0.0.1 --port "$PORT" --metrics \
    --max-resident-models 1 --log-level info >"$LOG" 2>&1 &
SRV=$!
cleanup() {
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    rm -f "$LOG" "$SAMPLER"
}
trap cleanup EXIT
for _ in $(seq 1 60); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "$BASE/health" >/dev/null || { echo -e "${RED}FAIL${NC}: server never came up"; exit 1; }

cache_bytes() {  # one sample of mlx_serve:mlx_cache_bytes, empty when --metrics is off
    # --max-time 2: a sample in flight while the inference thread holds the
    # registry mutex must not wedge the sampler loop for the whole window.
    curl -sf --max-time 2 "$BASE/metrics" 2>/dev/null | awk '/^mlx_serve:mlx_cache_bytes/ {print $2}'
}
wait_state() {  # wait until the model's state field equals $2
    for _ in $(seq 1 240); do
        curl -sf "$BASE/v1/models" 2>/dev/null | \
            grep -o "\"id\":\"$1\"[^}]*\"state\":\"[a-z]*\"" | grep -q "\"state\":\"$2\"" && return 0
        sleep 1
    done
    return 1
}
chat() {
    curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":8}"
}

echo "  [1] load the victim ($VICTIM, ${V_GB}G on disk)"
[ "$(chat "$VICTIM")" = "200" ] || { echo -e "${RED}FAIL${NC}: victim chat did not return 200"; exit 1; }
wait_state "$VICTIM" ready || { echo -e "${RED}FAIL${NC}: victim never reached ready"; exit 1; }

echo "  [2] load the next model ($NEXT): make-room evicts the victim, then the preflight runs"
( END=$((SECONDS + 180))
  while [ $SECONDS -lt $END ]; do
      B=$(cache_bytes); [ -n "$B" ] && echo "$SECONDS $B" >> "$SAMPLER"
      sleep 0.5
  done ) &
SAMPLER_PID=$!

CODE=$(chat "$NEXT")
kill $SAMPLER_PID 2>/dev/null
WORST=$(sort -n "$SAMPLER" 2>/dev/null | awk '{if ($2+0 > w) w = $2+0} END {print int(w/1048576)}')
echo "  pool samples across the evict→load window (max ${WORST:-0} MB, bar < ${CACHE_LIMIT_MB} MB):"
sort -n "$SAMPLER" | awk '{printf "    t=%ss pool=%.0f MB\n", $1, $2/1048576}' | tail -12

RED_REASON=""
if [ "$CODE" != "200" ]; then
    grep -E "evicting model id=|Insufficient memory" "$LOG" | tail -3 | sed 's/^/    log: /'
    RED_REASON="the load after a make-room eviction returned $CODE (want 200;"
    RED_REASON="$RED_REASON the victim's bytes were still parked in the allocator pool the preflight reads)"
elif ! grep -q "evicting model id=" "$LOG"; then
    RED_REASON="200 but no eviction line — make-room never ran, the run proves nothing"
elif [ "${WORST:-0}" -ge "$CACHE_LIMIT_MB" ]; then
    RED_REASON="mlx_cache_bytes held ${WORST} MB across the window (bar < ${CACHE_LIMIT_MB} MB):"
    RED_REASON="$RED_REASON the registry books zeroed but the victim's pool never left the process"
fi
if [ -n "$RED_REASON" ]; then
    echo -e "${RED}FAIL${NC}: $RED_REASON"
    exit 1
fi
wait_state "$NEXT" ready || { echo -e "${RED}FAIL${NC}: next model never reached ready"; exit 1; }
echo -e "${GREEN}PASS${NC}: the load after a make-room eviction served (200) and the pool left the process"
