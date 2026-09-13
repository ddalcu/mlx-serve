#!/bin/bash
# The hot-cache byte budget follows the machine's residency (issue #364).
#
# The budget used to be clamped ONCE at load against everything resident and
# never revisited: a model loaded beside a large one kept a ~0 budget for
# life, and unloading the neighbour did not give it back. Model A boots with
# no --prefix-cache-mem (budget = headroom, so every change is visible), B
# loads beside it, then unloads (`--prefix-cache-mem 0` = the uncapped ask):
#   [1] A's budget is revised DOWN when B loads, and A's warm turn still hits
#   [2] A's budget is revised UP once B is unloaded and its pages are back
#
# Usage: ./tests/test_prefix_cache_budget_revisit.sh [port]
set -u
PORT="${1:-11441}"
BASE="http://127.0.0.1:$PORT"
MODEL_A="$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit"
MODEL_B="$HOME/.mlx-serve/models/mlx-community/gemma-4-e2b-it-4bit"
BIN="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
for m in "$MODEL_A" "$MODEL_B"; do
    [ -d "$m" ] || { echo -e "${YELLOW}SKIP${NC} test_prefix_cache_budget_revisit: $m not found"; exit 0; }
done
[ -x "$BIN" ] || { echo -e "${RED}FAIL${NC} $BIN missing"; exit 1; }

LOG=$(mktemp)
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 0.5
"$BIN" --serve --model "$MODEL_A" --model-dir "$HOME/.mlx-serve/models" --host 127.0.0.1 --port "$PORT" \
    --ctx-size 8192 --prefix-cache-mem 0 --prefix-cache-disk off --log-level info >"$LOG" 2>&1 &
SRV=$!
cleanup() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT
for _ in $(seq 1 240); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || break
    sleep 0.5
done
curl -sf "$BASE/health" >/dev/null || { echo -e "${RED}FAIL${NC} server never became healthy"; tail -5 "$LOG"; exit 1; }

post() { curl -s -o /dev/null -w '%{http_code}' --max-time 300 -X POST "$BASE/v1/$1" -H 'Content-Type: application/json' -d "$2"; }
chat() { # chat <model path> -> cached_tokens
    python3 -c "
import json,sys,urllib.request
b={'model':sys.argv[1],'messages':[{'role':'system','content':'You are terse. '+'Rule %d: be brief. '*120 % tuple(range(120))},{'role':'user','content':'Say hi.'}],'max_tokens':4,'temperature':0}
r=json.load(urllib.request.urlopen(urllib.request.Request(sys.argv[2]+'/v1/chat/completions',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})))
print(r['usage']['prompt_tokens_details']['cached_tokens'])" "$1" "$BASE"
}
budget_line() { grep -oE "resident=[0-9.]+ / [0-9]+" "$LOG" | tail -1 | sed 's|.*/ ||'; }

FAIL=0
_=$(chat "$MODEL_A"); _=$(chat "$MODEL_A")
B0=$(budget_line)
echo "  A budget at load: ${B0} MB"

[ "$(post load-model "{\"model\":\"$MODEL_B\"}")" = "200" ] || { echo -e "${RED}FAIL${NC} load B"; tail -5 "$LOG"; exit 1; }
sleep 1
B1=$(grep -oE "\[hot-cache\] budget revised [0-9]+ -> [0-9]+ MB" "$LOG" | tail -1 | grep -oE "[0-9]+ MB$" | cut -d' ' -f1)
WARM=$(chat "$MODEL_A")
if [ -n "$B1" ] && [ "$B1" -lt "$B0" ] && [ "$WARM" -gt 100 ]; then
    echo -e "${GREEN}PASS${NC} [1] A's budget revised down when B loaded (${B0} -> ${B1} MB), A still warm (cached_tokens=$WARM)"
else
    echo -e "${RED}FAIL${NC} [1] expected a downward revision and a warm A: B0=$B0 B1='$B1' cached=$WARM"; FAIL=1
fi

[ "$(post unload-model "{\"model\":\"$MODEL_B\"}")" = "200" ] || { echo -e "${RED}FAIL${NC} unload B"; FAIL=1; }
# The OS hands B's pages back lazily; the revise repeats before each prefill for 10 s.
sleep 5; _=$(chat "$MODEL_A")
B2=$(grep -oE "\[hot-cache\] budget revised [0-9]+ -> [0-9]+ MB" "$LOG" | tail -1 | grep -oE "[0-9]+ MB$" | cut -d' ' -f1)
if [ -n "$B2" ] && [ -n "$B1" ] && [ "$B2" -gt "$B1" ]; then
    echo -e "${GREEN}PASS${NC} [2] A's budget revised up when B unloaded (${B1} -> ${B2} MB)"
else
    echo -e "${RED}FAIL${NC} [2] expected an upward revision: B1='$B1' B2='$B2'"; grep "hot-cache\] budget" "$LOG"; FAIL=1
fi
[ "$FAIL" = 0 ] && echo -e "${GREEN}ALL PASS${NC} test_prefix_cache_budget_revisit" || { grep -E "hot-cache\] budget|registry\]" "$LOG" | tail -12; exit 1; }
