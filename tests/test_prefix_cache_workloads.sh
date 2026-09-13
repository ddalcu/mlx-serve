#!/bin/bash
# Workload-fair hot-cache eviction (issue #378).
#
# One model, two workloads sharing a 4-entry hot prefix cache: a conversation
# under a system prompt, and a sweep of six distinct documents. Plain LRU let
# the sweep evict the conversation every time. Eviction now picks the LRU of
# the workload key holding the most entries (`server.requestCacheKey`:
# prompt_cache_key > metadata.user_id > system-prompt hash > anonymous), so the
# sweep only evicts its own documents.
#
#   [1] conversation turn 2 is warm (cached_tokens > 1000: the system prompt, not a 2-token BOS match)
#   [2] six docs with prompt_cache_key:"batch"; turn 3 is still warm
#   [3] every eviction during that sweep names the ONE batch key
#   [4] control: six unkeyed, system-less docs (anonymous group); turn 4 still warm
#   [5] the control sweep evicts key-0 entries (it pays for itself)
# A turn may evict the conversation's own stale earlier entry, and a tie between
# groups falls back to plain LRU: both are the policy, so only the NEWEST
# conversation entry ([2], [4]) is the bar.
#
# Usage: ./tests/test_prefix_cache_workloads.sh [/path/to/model] [port]

set -e

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
PORT="${2:-11433}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_prefix_cache_workloads: $MODEL not found."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

LOGFILE=$(mktemp)
echo "  starting server (--prefix-cache-entries 4)..."
"$BINARY" --model "$MODEL" --serve --port "$PORT" --host 127.0.0.1 --prefix-cache-entries 4 --prefix-cache-disk off --log-level info ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    cp "$LOGFILE" "$HOME/claude-tmp/pc378/server.log" 2>/dev/null; rm -f "$LOGFILE"
}
trap cleanup EXIT

up=0
for i in $(seq 1 90); do
    curl -s -f "$BASE/health" > /dev/null 2>&1 && { up=1; break; }
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} server did not become healthy"; tail -30 "$LOGFILE"; exit 1
fi

# ~2k tokens of system prompt so the conversation is a real entry.
SYSTEM=$(python3 -c "print('You are a careful assistant for the Orion project. Rule %d: answer briefly. ' * 220 % tuple(range(220)))")
HISTORY="[]"

# $1 = user text; appends the turn to HISTORY, prints cached_tokens.
conv_turn() {
    local body resp reply
    body=$(python3 -c "
import json,sys
hist=json.loads(sys.argv[1]); hist.append({'role':'user','content':sys.argv[2]})
print(json.dumps({'messages':[{'role':'system','content':sys.argv[3]}]+hist,'max_tokens':12,'temperature':0}))
" "$HISTORY" "$1" "$SYSTEM")
    resp=$(curl -s -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
    reply=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
    HISTORY=$(python3 -c "
import json,sys
hist=json.loads(sys.argv[1]); hist.append({'role':'user','content':sys.argv[2]}); hist.append({'role':'assistant','content':sys.argv[3]})
print(json.dumps(hist))" "$HISTORY" "$1" "$reply")
    echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['prompt_tokens_details']['cached_tokens'])"
}

# $1 = doc index, $2 = extra JSON fields (e.g. '"prompt_cache_key":"batch",')
sweep_doc() {
    local doc
    doc=$(python3 -c "print(('Document $1 paragraph %d about topic $1. ' * 8) % tuple(range(8)) * 20)")
    python3 -c "
import json,sys
print(json.dumps({'messages':[{'role':'user','content':'Summarize in three words: '+sys.argv[1]}],'max_tokens':8,'temperature':0}))" "$doc" \
      | sed "s/^{/{$2/" \
      | curl -s -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- > /dev/null
}

# $1 = log line count at the start of the phase.
evicted_keys_since() {
    tail -n +"$(($1 + 1))" "$LOGFILE" | grep -o 'evicted LRU entry ([^;]*; key=[0-9a-f]*' | sed 's/.*key=//' | sort -u
}

FAIL=0
c1=$(conv_turn "Say hello.")
c2=$(conv_turn "What is rule 3?")
if [ "$c2" -gt 1000 ]; then echo -e "${GREEN}PASS${NC} [1] turn 2 warm (cached_tokens=$c2)"; else echo -e "${RED}FAIL${NC} [1] turn 2 cold (cached_tokens=$c2)"; FAIL=1; fi

mark=$(wc -l < "$LOGFILE")
for i in 1 2 3 4 5 6; do sweep_doc "$i" '"prompt_cache_key":"batch",'; done
keys=$(evicted_keys_since "$mark")
c3=$(conv_turn "And rule 7?")
if [ "$c3" -gt 1000 ]; then echo -e "${GREEN}PASS${NC} [2] turn 3 warm after a keyed sweep (cached_tokens=$c3)"; else echo -e "${RED}FAIL${NC} [2] keyed sweep evicted the conversation (cached_tokens=$c3)"; FAIL=1; fi

nkeys=$(echo "$keys" | grep -c . || true)
if [ "$nkeys" = "1" ] && [ "$keys" != "0" ]; then
    echo -e "${GREEN}PASS${NC} [3] every eviction names the batch key ($keys)"
else
    echo -e "${RED}FAIL${NC} [3] expected one non-anonymous evicted key, got: $(echo $keys)"; grep 'evicted LRU' "$LOGFILE"; FAIL=1
fi
BATCH_KEY="$keys"

mark=$(wc -l < "$LOGFILE")
for i in 7 8 9 10 11 12; do sweep_doc "$i" ''; done
anon_evicted=$(evicted_keys_since "$mark" | grep -c -x 0 || true)
c4=$(conv_turn "Rule 11?")
if [ "$c4" -gt 1000 ]; then echo -e "${GREEN}PASS${NC} [4] turn 4 warm after an anonymous sweep (cached_tokens=$c4)"; else echo -e "${RED}FAIL${NC} [4] anonymous sweep evicted the conversation (cached_tokens=$c4)"; FAIL=1; fi

if [ "$anon_evicted" = "1" ]; then
    echo -e "${GREEN}PASS${NC} [5] the anonymous sweep evicts its own documents"
else
    echo -e "${RED}FAIL${NC} [5] no key=0 eviction during the anonymous sweep"; grep 'evicted LRU' "$LOGFILE"; FAIL=1
fi

[ "$FAIL" = "0" ] && echo -e "${GREEN}ALL PASS${NC} test_prefix_cache_workloads" || { tail -40 "$LOGFILE"; exit 1; }
