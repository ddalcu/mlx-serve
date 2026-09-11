#!/bin/bash
# --idle-evict-secs: a model nobody is using leaves residency, and the next
# request cold-loads it back.
#
# The reload this makes routine is also what the test is for. Three endpoints
# read an entry's RETAINED CPU state (config, chat template) without holding a
# refcount — /api/tags, /api/show, and the pre-load text-gen gate on a chat
# request — and a reload frees that state. So each cycle fires all three
# concurrently with the reload. Residency is read back from /api/ps rather than
# trusting the `state` field: "unloaded" with the memory still held is the
# failure this flag exists to prevent.
#
# Usage: ./tests/test_idle_evict.sh [root] [port]

set -e

ROOT="${1:-$HOME/.mlx-serve/models}"
PORT="${2:-8098}"
BASE="http://127.0.0.1:$PORT"
CYCLES="${IDLE_EVICT_CYCLES:-6}"
# Resident bytes may not grow across cycles that all end with NOTHING loaded:
# each reload replaces the previous generation of CPU state and must free it.
# Measured from cycle 2, not cycle 1 — the first concurrent cycle pays a
# one-time warm-up (prefix cache, MLX buffer pool, the OS returning freed pages
# lazily) worth hundreds of MB. Cycles 2..6 are flat to single-digit MB.
RSS_GROWTH_LIMIT_MB="${IDLE_EVICT_RSS_LIMIT_MB:-40}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

if [ ! -d "$ROOT" ]; then
    echo -e "${YELLOW}SKIP${NC} idle-evict: $ROOT not found."
    exit 0
fi
source "$(dirname "$0")/_lib_supported_models.sh"
MODELS=()
while IFS= read -r m; do MODELS+=("$m"); done < <(list_supported_models "$ROOT" 1)
if [ "${#MODELS[@]}" -lt 1 ]; then
    echo -e "${YELLOW}SKIP${NC} idle-evict: no supported model under $ROOT."
    exit 0
fi
M="${MODELS[0]}"
echo "  using $M"

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1
LOGFILE=$(mktemp)
# 2s window: idleEvictTickMs floors the sweep at 1s, so eviction lands ~3s after
# the last request instead of making the test wait out a realistic window.
"$BINARY" --model-dir "$ROOT" --serve --port "$PORT" --idle-evict-secs 2 \
    ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
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

chat() {  # prints the HTTP status
    curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":4}"
}
# Resident entries only — /api/ps lists `.ready` models with their bytes.
resident_count() { curl -fs "$BASE/api/ps" | grep -o '"name"' | wc -l | tr -d ' '; }
server_rss_mb() { ps -o rss= -p $SERVER_PID | awk '{printf "%d", $1/1024}'; }

wait_unloaded() {  # 0 = nothing resident within the timeout
    for _ in $(seq 1 40); do
        [ "$(resident_count)" = "0" ] && return 0
        sleep 0.5
    done
    return 1
}

echo "  cycle 1: load, go idle, evict"
[ "$(chat)" = "200" ] || { echo -e "${RED}FAIL${NC} first request did not return 200"; FAIL=1; }
[ "$(resident_count)" = "1" ] || { echo -e "${RED}FAIL${NC} model not resident after a request"; FAIL=1; }
if wait_unloaded; then
    grep -q "idle-evicting model id=" "$LOGFILE" || {
        echo -e "${RED}FAIL${NC} residency dropped but the sweep never logged it"; FAIL=1; }
else
    echo -e "${RED}FAIL${NC} model still resident 20s into a 2s idle window"; FAIL=1
fi
echo "    resident bytes released; server RSS $(server_rss_mb) MB"
BASE_RSS=""

# A status poll must not be a load. The app's tray polls /props every 3s and the
# OpenCode plugin polls it too, so a /props that cold-loads takes the memory
# straight back and eviction looks broken.
curl -fs "$BASE/props" >/dev/null || { echo -e "${RED}FAIL${NC} /props did not answer with nothing loaded"; FAIL=1; }
curl -fs "$BASE/metrics" >/dev/null 2>&1 || true
curl -fs "$BASE/v1/models" >/dev/null || true
sleep 2
if [ "$(resident_count)" = "0" ]; then
    echo "  status polls left it unloaded"
else
    echo -e "${RED}FAIL${NC} a status poll reloaded the model — eviction cannot win against it"; FAIL=1
fi

# Each cycle reloads the model while three unrefcounted readers of its retained
# CPU state run against it.
for i in $(seq 2 "$CYCLES"); do
    PIDS=()
    for _ in 1 2 3; do ( chat >/dev/null ) & PIDS+=($!); done
    for _ in 1 2 3 4 5 6; do
        ( curl -fs "$BASE/api/tags" >/dev/null ) & PIDS+=($!)
        ( curl -fs -X POST "$BASE/api/show" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$M\"}" >/dev/null ) & PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait "$p" || { echo -e "${RED}FAIL${NC} cycle $i: a concurrent request failed"; FAIL=1; }; done
    curl -fs "$BASE/health" >/dev/null || { echo -e "${RED}FAIL${NC} cycle $i: server died"; FAIL=1; break; }
    wait_unloaded || { echo -e "${RED}FAIL${NC} cycle $i: never went idle"; FAIL=1; break; }
    CYCLE_RSS=$(server_rss_mb)
    [ -n "$BASE_RSS" ] || BASE_RSS="$CYCLE_RSS"   # cycle 2 = the warm baseline
    echo "    cycle $i ok — RSS ${CYCLE_RSS} MB"
done

if [ "$FAIL" = "0" ] && [ -n "$BASE_RSS" ]; then
    END_RSS=$(server_rss_mb)
    GROWTH=$((END_RSS - BASE_RSS))
    echo "  RSS across cycles 2..$CYCLES, nothing resident at either end: ${BASE_RSS} → ${END_RSS} MB (+${GROWTH})"
    if [ "$GROWTH" -gt "$RSS_GROWTH_LIMIT_MB" ]; then
        echo -e "${RED}FAIL${NC} +${GROWTH} MB with nothing resident — a reload is orphaning its CPU state"
        FAIL=1
    fi
fi

# Default off: the same binary with no flag must keep the model resident.
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
"$BINARY" --model-dir "$ROOT" --serve --port "$PORT" ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 30); do curl -fs "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
[ "$(chat)" = "200" ] || { echo -e "${RED}FAIL${NC} control server did not serve"; FAIL=1; }
sleep 8
if [ "$(resident_count)" = "1" ]; then
    echo "  default off: still resident after 8s idle"
else
    echo -e "${RED}FAIL${NC} model evicted with no --idle-evict-secs — the flag is not the gate"; FAIL=1
fi

if [ "$FAIL" = "0" ]; then
    echo -e "${GREEN}PASS${NC} idle-evict: evicts when idle, reloads under concurrent readers, frees what it replaces"
else
    echo -e "${RED}FAIL${NC} idle-evict"
    echo "--- server log (tail) ---"; tail -40 "$LOGFILE"
fi
exit $FAIL
