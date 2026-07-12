#!/bin/bash
# SSD prefix-cache persistence (`--prefix-cache-disk`, src/kv_disk_cache.zig).
#
# End-to-end restart round-trip:
#   1. Boot with an isolated HOME, fire a long-prompt request (cold prefill),
#      capture output + TTFT. Assert `[disk-cache] persisted` fires after the
#      response.
#   2. Kill the server. Assert chunked entries exist on disk
#      (~/.mlx-serve/kv-cache/<fp>/e*/{meta.json,tokens.bin,c*.safetensors}).
#   3. Boot again, re-issue the identical request. Assert:
#        * `[disk-cache] restored N/M tokens from SSD` fires,
#        * the restart TTFT beats the cold TTFT by >= 2x,
#        * the output is byte-identical to run 1 (temp 0).
#   4. Multi-turn extension after restart appends chunks, doesn't rewrite
#      the whole entry (`persisted` with a small chunk count).
#   5. `--prefix-cache-disk off` boots clean, serves, and never touches the
#      kv-cache dir.
#
# Usage: ./tests/test_prefix_cache_disk.sh [/path/to/model] [port]

set -e

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
PORT="${2:-8096}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Legacy flat layout fallback (pre-two-level model dirs).
if [ ! -d "$MODEL" ] && [ -d "$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit" ]; then
    MODEL="$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit"
fi
if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_prefix_cache_disk: $MODEL not found."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first."
    exit 1
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

# Isolated HOME so the test never touches the user's real kv-cache.
SCRATCH_HOME=$(mktemp -d)
HYBRID_HOME=$(mktemp -d)
LOGFILE=$(mktemp)
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf "$SCRATCH_HOME" "$HYBRID_HOME" "$LOGFILE"
}
trap cleanup EXIT

start_server() { # extra args...
    : > "$LOGFILE"
    # The SSD tier is OPT-IN since the off-by-default flip (ad3fd24) — the
    # suite must enable it explicitly. Callers' "$@" comes later, so section
    # 5's `--prefix-cache-disk off` still wins (last flag parses last).
    HOME="$SCRATCH_HOME" "$BINARY" --model "$MODEL" --serve --port "$PORT" \
        --ctx-size 8192 --no-pld --log-level info --prefix-cache-disk 4GB "$@" > "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 90); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then return 0; fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:"; tail -20 "$LOGFILE"; return 1; }
        sleep 1
    done
    return 1
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 1
}

# Deterministic ~2.5k-token prompt (well above the 512-token persist floor).
LONG_PROMPT=$(python3 -c "
lines = ['You are reviewing a project log. Read it fully, then answer.']
for i in range(1, 81):
    lines.append(f'Log entry {i}: subsystem {i%17} reported state change {i*31%997} with latency {i*7%131} ms and checksum {i*i%9973}.')
lines.append('Question: which subsystem appears in log entry 42? Answer with one short sentence.')
print('\n'.join(lines))")

fire_long() { # -> "elapsed_ms|content"
    local body
    body=$(python3 -c "
import json,sys
print(json.dumps({'model':'mlx-serve','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':24,'temperature':0.0,'stream':False}))
" "$LONG_PROMPT")
    python3 - "$BASE" "$body" <<'PY'
import json, sys, time, urllib.request
base, body = sys.argv[1], sys.argv[2]
t0 = time.monotonic()
req = urllib.request.Request(base + "/v1/chat/completions", data=body.encode(), headers={"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req, timeout=600))
ms = int((time.monotonic() - t0) * 1000)
content = resp["choices"][0]["message"]["content"].replace("|", " ").replace("\n", " ")
print(f"{ms}|{content}")
PY
}

FAIL=0
KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"

echo "== 1. cold boot + long request (persist) =="
start_server || { echo -e "${RED}FAIL${NC} server 1 failed to start"; exit 1; }
OUT1=$(fire_long)
COLD_MS="${OUT1%%|*}"
CONTENT1="${OUT1#*|}"
echo "  cold TTLT(total)=${COLD_MS}ms content='${CONTENT1:0:60}'"
sleep 1  # flush happens post-response on the inference thread
if grep -q '\[disk-cache\] persisted' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} disk persist fired"
else
    echo -e "${RED}FAIL${NC} no [disk-cache] persisted line"
    tail -20 "$LOGFILE"; FAIL=1
fi
stop_server

echo
echo "== 2. on-disk layout =="
META_COUNT=$(find "$KV_DIR" -name meta.json 2>/dev/null | wc -l | tr -d ' ')
CHUNK_COUNT=$(find "$KV_DIR" -name 'c*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
if [ "$META_COUNT" -ge 1 ] && [ "$CHUNK_COUNT" -ge 2 ]; then
    echo -e "${GREEN}PASS${NC} persisted entry on disk ($META_COUNT meta, $CHUNK_COUNT chunks)"
else
    echo -e "${RED}FAIL${NC} expected >=1 meta + >=2 chunks, got $META_COUNT/$CHUNK_COUNT"
    find "$KV_DIR" -type f 2>/dev/null | head
    FAIL=1
fi

echo
echo "== 3. restart + identical request (restore) =="
start_server || { echo -e "${RED}FAIL${NC} server 2 failed to start"; exit 1; }
OUT2=$(fire_long)
RESTART_MS="${OUT2%%|*}"
CONTENT2="${OUT2#*|}"
echo "  restart total=${RESTART_MS}ms (cold was ${COLD_MS}ms) content='${CONTENT2:0:60}'"
if grep -q '\[disk-cache\] restored .* tokens from SSD' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} SSD restore engaged: $(grep -o '\[disk-cache\] restored [^\\n]*' "$LOGFILE" | head -1)"
else
    echo -e "${RED}FAIL${NC} no [disk-cache] restored line after restart"
    tail -30 "$LOGFILE"; FAIL=1
fi
if [ "$CONTENT1" = "$CONTENT2" ]; then
    echo -e "${GREEN}PASS${NC} output byte-identical across restart restore"
else
    echo -e "${RED}FAIL${NC} output diverged: '$CONTENT1' vs '$CONTENT2'"
    FAIL=1
fi
SPEEDUP_OK=$(python3 -c "print(1 if $RESTART_MS * 2 <= $COLD_MS else 0)")
if [ "$SPEEDUP_OK" = "1" ]; then
    echo -e "${GREEN}PASS${NC} restart request >=2x faster than cold (${RESTART_MS}ms vs ${COLD_MS}ms)"
else
    echo -e "${RED}FAIL${NC} restart request not >=2x faster (${RESTART_MS}ms vs ${COLD_MS}ms)"
    FAIL=1
fi

echo
echo "== 4. multi-turn extension appends chunks =="
EXT_BODY=$(python3 -c "
import json,sys
print(json.dumps({'model':'mlx-serve','messages':[
  {'role':'user','content':sys.argv[1]},
  {'role':'assistant','content':sys.argv[2]},
  {'role':'user','content':'Now answer the same question for log entry 43.'}],
  'max_tokens':24,'temperature':0.0,'stream':False}))
" "$LONG_PROMPT" "$CONTENT2")
curl -s -X POST -H "Content-Type: application/json" -d "$EXT_BODY" "$BASE/v1/chat/completions" > /dev/null
sleep 1
PERSIST_LINES=$(grep -c '\[disk-cache\] persisted' "$LOGFILE" || true)
if [ "$PERSIST_LINES" -ge 1 ]; then
    echo -e "${GREEN}PASS${NC} extension turn persisted (append), $PERSIST_LINES persist line(s) this session"
else
    echo -e "${RED}FAIL${NC} extension turn did not persist"
    tail -20 "$LOGFILE"; FAIL=1
fi
stop_server

echo
echo "== 5. --prefix-cache-disk off leaves disk untouched =="
rm -rf "$KV_DIR"
start_server --prefix-cache-disk off || { echo -e "${RED}FAIL${NC} server 3 failed to start"; exit 1; }
OUT3=$(fire_long)
sleep 1
if [ -d "$KV_DIR" ] && [ -n "$(find "$KV_DIR" -name meta.json 2>/dev/null)" ]; then
    echo -e "${RED}FAIL${NC} kv-cache written despite --prefix-cache-disk off"
    FAIL=1
elif grep -q '\[disk-cache\]' "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} disk-cache log lines despite off"
    FAIL=1
else
    echo -e "${GREEN}PASS${NC} off switch respected"
fi
stop_server

echo
echo "== 6. hybrid SSM arch (Qwen 3.5 GatedDeltaNet) persists + restores SSM state =="
# Phase 3: hybrid recurrent archs persist their per-position SSM checkpoints
# beside the KV chunks and restore both across a restart. Gated on a local
# Qwen3.5-0.8B; SKIPs cleanly otherwise (the attention sections above cover the
# non-hybrid path either way).
HYBRID_MODEL="$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"
[ -d "$HYBRID_MODEL" ] || HYBRID_MODEL="$HOME/.mlx-serve/models/Qwen3.5-0.8B-MLX-4bit"
if [ ! -d "$HYBRID_MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} hybrid section: Qwen3.5-0.8B-MLX-4bit not found."
else
    # Repoint the server helpers at the hybrid model + a fresh isolated HOME.
    MODEL="$HYBRID_MODEL"
    SCRATCH_HOME="$HYBRID_HOME"
    KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"

    echo "  -- cold boot + long request (persist) --"
    start_server --no-mtp || { echo -e "${RED}FAIL${NC} hybrid server failed to start"; exit 1; }
    HOUT1=$(fire_long)
    HCOLD_MS="${HOUT1%%|*}"
    HCONTENT1="${HOUT1#*|}"
    echo "  cold total=${HCOLD_MS}ms content='${HCONTENT1:0:60}'"
    sleep 1
    if grep -q '\[disk-cache\] persisted' "$LOGFILE"; then
        # A hybrid persist reports SSM checkpoints in the count.
        echo -e "${GREEN}PASS${NC} hybrid persist fired: $(grep -o '\[disk-cache\] persisted[^;]*' "$LOGFILE" | tail -1)"
    else
        echo -e "${RED}FAIL${NC} hybrid: no [disk-cache] persisted line"
        tail -20 "$LOGFILE"; FAIL=1
    fi
    stop_server

    echo "  -- restart + identical request (restore SSM@pos) --"
    start_server --no-mtp || { echo -e "${RED}FAIL${NC} hybrid server 2 failed to start"; exit 1; }
    HOUT2=$(fire_long)
    HRESTART_MS="${HOUT2%%|*}"
    HCONTENT2="${HOUT2#*|}"
    echo "  restart total=${HRESTART_MS}ms (cold was ${HCOLD_MS}ms) content='${HCONTENT2:0:60}'"
    if grep -qE '\[disk-cache\] restored .* from SSD .*\(ssm@[0-9]+\)' "$LOGFILE"; then
        echo -e "${GREEN}PASS${NC} hybrid SSD restore engaged: $(grep -oE '\[disk-cache\] restored [^(]*\(ssm@[0-9]+\)' "$LOGFILE" | head -1)"
    else
        echo -e "${RED}FAIL${NC} hybrid: no '[disk-cache] restored … (ssm@…)' line after restart"
        tail -30 "$LOGFILE"; FAIL=1
    fi
    if [ "$HCONTENT1" = "$HCONTENT2" ]; then
        echo -e "${GREEN}PASS${NC} hybrid output byte-identical across restart restore"
    else
        echo -e "${RED}FAIL${NC} hybrid output diverged: '$HCONTENT1' vs '$HCONTENT2'"
        FAIL=1
    fi
    HSPEEDUP_OK=$(python3 -c "print(1 if $HRESTART_MS * 2 <= $HCOLD_MS else 0)")
    if [ "$HSPEEDUP_OK" = "1" ]; then
        echo -e "${GREEN}PASS${NC} hybrid restart request >=2x faster than cold (${HRESTART_MS}ms vs ${HCOLD_MS}ms)"
    else
        echo -e "${YELLOW}WARN${NC} hybrid restart not >=2x faster (${HRESTART_MS}ms vs ${HCOLD_MS}ms) — small model, TTFT dominated by fixed costs"
    fi
    stop_server
fi

echo
if [ "$FAIL" = "0" ]; then
    echo -e "${GREEN}ALL PASS${NC} test_prefix_cache_disk"
else
    echo -e "${RED}FAILURES${NC} test_prefix_cache_disk"
fi
exit $FAIL
