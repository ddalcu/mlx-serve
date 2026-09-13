#!/bin/bash
# Issue #390: a 1-token-tail commit that inherits checkpoints from a shared
# prefix must not poison later restores with QsaHistoryGap.
#
# Two conversations share one ~15k-token prefix. Conversation A grows past
# the prefix (donor bank sits above the shared position). Conversation B
# restores, prefills a 1-token tail at an odd position, commits (cps=null
# shape), then restores again. Every turn must 200; the log must not contain
# `prefill failed for slot: QsaHistoryGap`. The self-heal line is allowed
# and counted as WARN. After inherit, one request is posted TWICE verbatim so
# the 1-token tail / full-reuse shape actually occurs; the script fails if it
# never did.
#
# Warm and cold run sequentially on ONE port: warm phase, capture greedy
# texts, stop, wait for the port and for free+inactive+speculative pages,
# then boot `--prefix-cache-entries 0` and replay. Never two servers at once.
#
# Usage: ./tests/test_prefix_inherit_qsa.sh [/path/to/qwen4_exp] [port]
#
# Coordinator runs this. Do not start a model server from the agent.

set -e

MODEL="${1:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit}"
PORT="${2:-11490}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'
MEM_RECOVER_GAP_MB=10240
MEM_RECOVER_TIMEOUT_S=90

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_prefix_inherit_qsa: $MODEL not found."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

SERVER_PID=""
LOGFILE=""
COLD_LOG=""

free_mb() {
  vm_stat | awk '
    /page size of/         { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) ps = $i }
    /^Pages free:/         { gsub(/\./, "", $3); free = $3 }
    /^Pages inactive:/     { gsub(/\./, "", $3); inact = $3 }
    /^Pages speculative:/  { gsub(/\./, "", $3); spec = $3 }
    END { if (ps == "") ps = 16384; printf "%d", (free + inact + spec) * ps / 1048576 }'
}

wait_port_closed() {
    local i
    for i in $(seq 1 60); do
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || return 0
        sleep 1
    done
    return 1
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] || return 0
    local was="$SERVER_PID"
    kill "$was" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$was" 2>/dev/null || break; sleep 1; done
    kill -9 "$was" 2>/dev/null || true
    wait "$was" 2>/dev/null || true
    SERVER_PID=""
    wait_port_closed || true
}

wait_for_release() {
    local pid="$1"
    local baseline="$2"
    local floor=$((baseline - MEM_RECOVER_GAP_MB))
    local waited=0
    local now
    while [ "$waited" -lt "$MEM_RECOVER_TIMEOUT_S" ]; do
        now=$(free_mb)
        if ! kill -0 "$pid" 2>/dev/null && [ "$now" -ge "$floor" ]; then
            lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || {
                echo "  pack released after ${waited}s (free+inactive+speculative ${now} MB >= floor ${floor} MB)"
                return 0
            }
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "  WARN: memory did not recover within ${MEM_RECOVER_TIMEOUT_S}s (free+inactive+speculative $(free_mb) MB, floor ${floor} MB) - proceeding"
    wait_port_closed || echo "  WARN: port $PORT still listening"
    return 0
}

cleanup() {
    stop_server
    rm -f "$LOGFILE" "$COLD_LOG"
}
trap cleanup EXIT INT TERM

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
wait_port_closed || true

BASELINE_FREE_MB=$(free_mb)
echo "  baseline free+inactive+speculative before any boot: ${BASELINE_FREE_MB} MB"

start_server() {
    local log="$1"
    shift
    echo "  starting server $*..."
    "$BINARY" --model "$MODEL" --serve --port "$PORT" --host 127.0.0.1 \
        "$@" --log-level info ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$log" 2>&1 &
    SERVER_PID=$!
    local up=0
    local i
    for i in $(seq 1 180); do
        curl -s -f "$BASE/health" > /dev/null 2>&1 && { up=1; break; }
        sleep 1
    done
    if [ "$up" != "1" ]; then
        echo -e "${RED}FAIL${NC} server did not become healthy"; tail -40 "$log"; exit 1
    fi
}

LOGFILE=$(mktemp)
start_server "$LOGFILE" --prefix-cache-entries 8 --prefix-cache-mem 2048MB --prefix-cache-disk off

SYSTEM=$(python3 -c "print('You are a careful assistant for the Orion project. Rule %d: answer briefly. ' * 900 % tuple(range(900)))")

chat() {
    local hist_json="$1"
    local user="$2"
    python3 -c "
import json,sys,urllib.request
hist=json.loads(sys.argv[1]); hist.append({'role':'user','content':sys.argv[2]})
body=json.dumps({'messages':[{'role':'system','content':sys.argv[3]}]+hist,'max_tokens':24,'temperature':0})
req=urllib.request.Request('$BASE/v1/chat/completions', data=body.encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=600) as r:
    raw=r.read(); code=r.status
obj=json.loads(raw)
reply=obj['choices'][0]['message']['content']
hist.append({'role':'assistant','content':reply})
print(json.dumps({'code':code,'hist':hist}))
" "$hist_json" "$user" "$SYSTEM"
}

fail=0
A='[]'
B='[]'
turn() {
    local which="$1" user="$2"
    local out
    if [ "$which" = A ]; then
        out=$(chat "$A" "$user") || { echo -e "${RED}FAIL${NC} A request failed"; fail=1; return; }
        A=$(echo "$out" | python3 -c "import json,sys; o=json.load(sys.stdin); print(json.dumps(o['hist'])); raise SystemExit(0 if o['code']==200 else 1)") || { echo -e "${RED}FAIL${NC} A not 200"; fail=1; }
    else
        out=$(chat "$B" "$user") || { echo -e "${RED}FAIL${NC} B request failed"; fail=1; return; }
        B=$(echo "$out" | python3 -c "import json,sys; o=json.load(sys.stdin); print(json.dumps(o['hist'])); raise SystemExit(0 if o['code']==200 else 1)") || { echo -e "${RED}FAIL${NC} B not 200"; fail=1; }
    fi
}

turn A "Hello"
turn A "Say one word."
turn B "Hello"
turn B "Hi"

BODY=$(python3 -c "
import json,sys
hist=json.loads(sys.argv[1])
hist.append({'role':'user','content':sys.argv[2]})
print(json.dumps({'messages':[{'role':'system','content':sys.argv[3]}]+hist,'max_tokens':24,'temperature':0}))
" "$B" "Ping." "$SYSTEM")

post_body() {
    python3 -c "
import json,sys,urllib.request
body=sys.argv[1].encode()
base=sys.argv[2]
req=urllib.request.Request(base+'/v1/chat/completions', data=body, headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=600) as r:
    raw=r.read(); code=r.status
obj=json.loads(raw)
print(json.dumps({'code':code,'text':obj['choices'][0]['message']['content']}))
" "$1" "$2"
}

WARM1=$(post_body "$BODY" "$BASE") || { echo -e "${RED}FAIL${NC} verbatim request 1 failed"; fail=1; WARM1='{"code":0,"text":""}'; }
WARM2=$(post_body "$BODY" "$BASE") || { echo -e "${RED}FAIL${NC} verbatim request 2 failed"; fail=1; WARM2='{"code":0,"text":""}'; }
echo "$WARM1" | python3 -c "import json,sys; o=json.load(sys.stdin); raise SystemExit(0 if o['code']==200 else 1)" || fail=1
echo "$WARM2" | python3 -c "import json,sys; o=json.load(sys.stdin); raise SystemExit(0 if o['code']==200 else 1)" || fail=1

if python3 -c "
import re,sys
text=open(sys.argv[1]).read()
for m in re.finditer(r'reused ([0-9]+)/([0-9]+) tokens', text):
    a,b=int(m.group(1)),int(m.group(2))
    if b-a==1:
        raise SystemExit(0)
raise SystemExit(1)
" "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} 1-token prefill shape occurred"
else
    echo -e "${RED}FAIL${NC} never saw a reused N/(N+1) 1-token prefill line"
    grep -n "hot-cache" "$LOGFILE" | head -20
    fail=1
fi

if grep -q "prefill failed for slot: QsaHistoryGap" "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} log contains prefill failed for slot: QsaHistoryGap"
    grep -n "QsaHistoryGap" "$LOGFILE" | head -10
    fail=1
else
    echo -e "${GREEN}PASS${NC} no QsaHistoryGap prefill failure"
fi

heal=$(grep -c "restored entry failed the QSA history check — dropped, cold prefill" "$LOGFILE" || true)
if [ "$heal" -gt 0 ]; then
    echo -e "${YELLOW}WARN${NC} self-heal events: $heal"
else
    echo "  self-heal events: 0"
fi

WARM_PID="$SERVER_PID"
stop_server
wait_for_release "$WARM_PID" "$BASELINE_FREE_MB"

COLD_LOG=$(mktemp)
start_server "$COLD_LOG" --prefix-cache-entries 0 --prefix-cache-disk off

COLD1=$(post_body "$BODY" "$BASE") || { echo -e "${RED}FAIL${NC} cold request 1 failed"; fail=1; COLD1='{"code":0,"text":""}'; }
python3 -c "
import json,sys,urllib.request
warm=json.loads(sys.argv[1])['text']
cold=json.loads(sys.argv[2])['text']
base=sys.argv[3]
if not warm:
    print('empty warm vs non-empty cold')
    raise SystemExit(1)
def toks(text):
    body=json.dumps({'content': text}).encode()
    req=urllib.request.Request(base+'/tokenize', data=body, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())['tokens']
wt=toks(warm)[:32]
ct=toks(cold)[:32]
if wt != ct:
    print('warm/cold first-32-token mismatch')
    print(' warm', wt[:8], 'len', len(toks(warm)))
    print(' cold', ct[:8], 'len', len(toks(cold)))
    raise SystemExit(1)
" "$WARM2" "$COLD1" "$BASE" || { echo -e "${RED}FAIL${NC} warm greedy text != cold"; fail=1; }
if [ "$fail" -eq 0 ]; then
    echo -e "${GREEN}PASS${NC} warm greedy text matches cold (first 32 tokens)"
fi

if [ "$fail" -eq 0 ]; then
    echo -e "${GREEN}PASS${NC} test_prefix_inherit_qsa"
    exit 0
fi
exit 1
