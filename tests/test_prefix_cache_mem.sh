#!/bin/bash
# Wave 1.B — `--prefix-cache-mem` memory budget enforcement.
#
# Starts the server with a deliberately tiny prefix-cache memory budget and
# fires several long-prompt requests in sequence. Asserts:
#   * `[hot-cache] resident=X.XX / Y.YY MB` log lines appear and X.XX never
#     exceeds Y.YY.
#   * At least one `[hot-cache] evicted LRU entry (byte budget; …)` log line
#     fires once the resident bytes would exceed the budget.
#   * The server stays healthy through all requests.
#   * (#330) ONE growing conversation crossing the budget gets TRIMMED, not
#     flat-declined — a `trimmed oversized entry to N/…` line appears and a
#     later turn logs `reused M/…` with M >= N. From the outside a trim that
#     silently stops engaging looks exactly like a legit decline, so only the
#     log can pin it; the unit tests own the trim's correctness. A hybrid
#     whose single checkpoint outweighs the whole budget legitimately
#     declines (`skipped oversized entry`) — that arm WARNs instead.
#
# Tunables (env): SHORT_BUDGET_MB (default 64), N_REQ (default 5),
#                 PROMPT_REPEATS (default 200), GROW_TURNS (default 5),
#                 GROW_WORDS (default 2000).
#
# Usage: ./tests/test_prefix_cache_mem.sh [/path/to/model] [port]

set -e

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
PORT="${2:-8094}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

SHORT_BUDGET_MB="${SHORT_BUDGET_MB:-64}"
N_REQ="${N_REQ:-5}"
PROMPT_REPEATS="${PROMPT_REPEATS:-200}"
GROW_TURNS="${GROW_TURNS:-5}"
GROW_WORDS="${GROW_WORDS:-2000}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_prefix_cache_mem: $MODEL not found."
    exit 0
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing — not a valid model directory."
    exit 1
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first."
    exit 1
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

LOGFILE=$(mktemp)
echo "  starting server (--prefix-cache-entries 8 --prefix-cache-mem ${SHORT_BUDGET_MB}MB)..."
"$BINARY" --model "$MODEL" --serve --port "$PORT" \
    --prefix-cache-entries 8 --prefix-cache-mem "${SHORT_BUDGET_MB}MB" \
    ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOGFILE" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; rm -f "$LOGFILE"; }
trap cleanup EXIT

up=0
for i in $(seq 1 60); do
    if curl -s -f "$BASE/health" > /dev/null 2>&1; then up=1; break; fi
    sleep 1
done
if [ "$up" != "1" ]; then
    echo -e "${RED}FAIL${NC} server did not become healthy in 60s"
    tail -30 "$LOGFILE"
    exit 1
fi

# Build a long-ish prompt so the resulting KV footprint is non-trivial. We
# vary the prompt slightly per-request so each one creates a fresh cache
# entry instead of extending the same one.
build_prompt() {
    local salt="$1"
    python3 -c "
import sys
salt = '$salt'
filler = (' Word.' * $PROMPT_REPEATS).strip()
print(f'Conversation {salt}: {filler}\nReply with a one-sentence acknowledgement.')
"
}

fire_one() {
    local salt="$1"
    local prompt
    prompt=$(build_prompt "$salt")
    local body
    body=$(python3 -c "
import json,sys
print(json.dumps({
    'model': 'mlx-serve',
    'messages': [{'role':'user','content': sys.argv[1]}],
    'max_tokens': 24,
    'temperature': 0.0,
    'stream': False,
}))
" "$prompt")
    local resp
    resp=$(echo "$body" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    local content
    content=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null || echo "")
    if [ -z "$content" ] || [ "$content" = "None" ]; then
        echo -e "${RED}FAIL${NC} request salt=$salt: empty/null content"
        echo "  body: $(echo "$resp" | head -c 400)"
        return 1
    fi
    return 0
}

FAIL=0
echo
echo "== firing $N_REQ long-prompt requests (budget=${SHORT_BUDGET_MB}MB) =="
for i in $(seq 1 "$N_REQ"); do
    fire_one "v$i" || FAIL=1
    sleep 1
done

# Inspect the log. We want:
#   1. At least one `[hot-cache] resident=…` line.
#   2. None of those lines exceed the budget.
#   3. At least one byte-budget eviction line.
echo
echo "== verifying log =="
RESIDENT_LINES=$(grep -E '\[hot-cache\] resident=.* MB' "$LOGFILE" || true)
if [ -z "$RESIDENT_LINES" ]; then
    echo -e "${RED}FAIL${NC} no `[hot-cache] resident=…` lines emitted"
    tail -50 "$LOGFILE"
    exit 1
fi

MAX_OBSERVED=$(echo "$RESIDENT_LINES" | python3 -c "
import sys,re
mx = 0.0
for line in sys.stdin:
    m = re.search(r'resident=([0-9.]+)\s*/\s*([0-9.]+)\s*MB', line)
    if m:
        mx = max(mx, float(m.group(1)))
print(mx)
")
echo "  max resident observed: ${MAX_OBSERVED} MB / ${SHORT_BUDGET_MB} MB"

OK=$(python3 -c "print(1 if ${MAX_OBSERVED} <= ${SHORT_BUDGET_MB} else 0)")
if [ "$OK" != "1" ]; then
    echo -e "${RED}FAIL${NC} resident bytes exceeded budget (${MAX_OBSERVED} > ${SHORT_BUDGET_MB} MB)"
    FAIL=1
else
    echo -e "${GREEN}PASS${NC} resident bytes stayed within budget"
fi

if grep -q '\[hot-cache\] evicted LRU entry (byte budget' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} byte-budget eviction observed"
else
    # Not necessarily a failure if the budget was generous enough that count
    # cap dominated — but for a 64MB budget over $N_REQ long requests it
    # should fire. We surface as FAIL only when N_REQ >= 4 (likely overflow).
    if [ "$N_REQ" -ge 4 ]; then
        echo -e "${RED}FAIL${NC} expected at least one byte-budget eviction at budget=${SHORT_BUDGET_MB}MB across $N_REQ long requests"
        echo "  log tail:"
        tail -40 "$LOGFILE"
        FAIL=1
    else
        echo -e "${YELLOW}WARN${NC} no byte-budget eviction (budget may be too generous for $N_REQ requests)"
    fi
fi

# ── #330: one growing conversation past the budget must TRIM, not go dark ──
echo
echo "== #330: growing conversation past the budget (${GROW_TURNS} turns x ~${GROW_WORDS} words) =="
GROW_MARK=$(wc -l < "$LOGFILE")
GROW_OK=1
python3 - "$BASE" "$GROW_TURNS" "$GROW_WORDS" <<'EOF' || GROW_OK=0
import json, sys, urllib.request

base, turns, words = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
msgs = [{"role": "system", "content": "You are a terse assistant."}]
for t in range(1, turns + 1):
    filler = " ".join(f"entry {t}.{i} value {(i * 7) % 97}" for i in range(words // 4))
    msgs.append({"role": "user", "content": f"Turn {t} log:\n{filler}\nAcknowledge in one sentence."})
    body = json.dumps({"model": "mlx-serve", "messages": msgs,
                       "max_tokens": 16, "temperature": 0.0}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    content = resp["choices"][0]["message"].get("content") or ""
    if not content:
        print(f"turn {t}: empty content", file=sys.stderr)
        sys.exit(1)
    msgs.append({"role": "assistant", "content": content})
    u = resp.get("usage", {})
    print(f"  turn {t}: prompt={u.get('prompt_tokens')} cached={u.get('prompt_tokens_details', {}).get('cached_tokens')}")
EOF
if [ "$GROW_OK" != "1" ]; then
    echo -e "${RED}FAIL${NC} growing conversation request failed"
    FAIL=1
fi

GROW_LOG=$(tail -n "+$((GROW_MARK + 1))" "$LOGFILE")
TRIM_LEN=$(echo "$GROW_LOG" | python3 -c "
import sys, re
best = 0
for line in sys.stdin:
    m = re.search(r'trimmed oversized entry to (\d+)/\d+ tokens', line)
    if m:
        best = max(best, int(m.group(1)))
print(best)
")
if [ "$TRIM_LEN" -gt 0 ]; then
    MAX_REUSED=$(echo "$GROW_LOG" | python3 -c "
import sys, re
best = 0
for line in sys.stdin:
    m = re.search(r'\[hot-cache\] reused (\d+)/\d+ tokens', line)
    if m:
        best = max(best, int(m.group(1)))
print(best)
")
    if [ "$MAX_REUSED" -ge "$TRIM_LEN" ]; then
        echo -e "${GREEN}PASS${NC} oversized entry trimmed to ${TRIM_LEN} tokens and later reused ${MAX_REUSED}"
    else
        echo -e "${RED}FAIL${NC} trimmed to ${TRIM_LEN} tokens but best later reuse was ${MAX_REUSED} — the trimmed entry never served"
        echo "$GROW_LOG" | grep -E '\[hot-cache\]' | tail -20
        FAIL=1
    fi
elif echo "$GROW_LOG" | grep -q '\[hot-cache\] skipped oversized entry'; then
    # Only a HYBRID can decline legitimately here (its smallest restorable
    # checkpoint alone can outweigh the budget — checkpoint stride = prefill
    # chunk). Hybrid entries are identifiable by the `; ssm` clause their
    # eviction lines carry; on a plain-attention arch a decline IS the #330
    # cliff and must fail.
    if grep -qE '\[hot-cache\] evicted LRU entry \(.*; ssm ' "$LOGFILE"; then
        echo -e "${YELLOW}WARN${NC} conversation crossed the budget but only declined — no restorable checkpoint fits ${SHORT_BUDGET_MB}MB on this hybrid"
    else
        echo -e "${RED}FAIL${NC} oversized conversation was declined instead of trimmed (#330 cliff) — every turn cold-prefills"
        echo "$GROW_LOG" | grep -E '\[hot-cache\]' | tail -20
        FAIL=1
    fi
else
    echo -e "${RED}FAIL${NC} conversation never crossed the ${SHORT_BUDGET_MB}MB budget — grow GROW_TURNS/GROW_WORDS; the #330 arm proved nothing"
    FAIL=1
fi

exit $FAIL
