#!/bin/bash
# Regression test for the persistent llama.cpp KV session across multiple turns.
#
# Reproduces the failure mode where the second turn of a chat — sharing some
# prefix with turn 1 but with a substantially larger new suffix — fails with:
#
#   init_batch: failed to prepare attention ubatches
#   decode: failed to find a memory slot for batch of size 512
#   [llama] session_sync rc=-1 err=llama_decode failed during prefill
#   [scheduler] prefill failed for slot: SessionSyncFailed
#
# That cascade is libllama refusing to allocate KV slots for the prefill batch
# on the second turn — it surfaces on Qwen3-family GGUFs (hybrid / sliding-window
# attention layers) when one `llama_context` is reused across requests via
# `seq_rm` + decode. The two-pronged fix:
#   1. `cp.swa_full = true` at session creation (full-size SWA cache, so
#      `seq_rm` actually exposes the trimmed slots to subsequent decodes).
#   2. `LlamaSession.syncWithFallback` in the scheduler — on a failed sync,
#      reset and retry once cold instead of surfacing a 500 to the client.
#
# This test exercises (1) directly: without `swa_full`, the second request
# returns a 500 with the SessionSyncFailed cascade in the server log. With
# the fix, both requests stream a normal SSE response with a terminator.
#
# Gated on the same `LLAMA_GGUF_MODEL` fixture as test_llama_gguf.sh:
#   LLAMA_GGUF_MODEL=/path/to/model.gguf ./tests/test_llama_persistent_session.sh [port]
set -uo pipefail

MODEL="${LLAMA_GGUF_MODEL:-}"
PORT="${1:-8124}"
BASE="http://127.0.0.1:$PORT"
BIN="./zig-out/bin/mlx-serve"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
PASS=0; FAIL=0

if [ -z "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_llama_persistent_session: set LLAMA_GGUF_MODEL=/path/to/model.gguf to run"
    exit 0
fi
if [ ! -f "$BIN" ]; then
    echo -e "${RED}ERROR${NC} $BIN not found — build with: zig build -Doptimize=ReleaseFast"
    exit 1
fi

ok()  { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
bad() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; [ -n "${2:-}" ] && echo "    $2"; }

LOG="$(mktemp)"
echo "→ starting mlx-serve on :$PORT with $MODEL (ctx=8192)"
"$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 8192 --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
SAVED_LOG="${SAVED_LOG:-}"
cleanup() {
    kill "$SERVER_PID" 2>/dev/null
    if [ -n "$SAVED_LOG" ]; then cp "$LOG" "$SAVED_LOG"; echo "(server log copied to $SAVED_LOG)"; fi
    rm -f "$LOG"
}
trap cleanup EXIT

# Wait for /health.
for i in $(seq 1 30); do
    if curl -fs --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"'; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo -e "${RED}server died${NC}"; tail -40 "$LOG"; exit 1; fi
    sleep 1
done

# Reproduce the user-reported failure pattern: an agentic chat where turn 2's
# history contains the previous assistant's multiple PARALLEL tool_calls plus
# matching tool-result messages — the shape that real Claude-Code-style agent
# loops produce. On the production failure the request had 6 tool messages
# (3 parallel tool_calls + 3 tool responses) on top of a 1.3KB system prompt
# and 3.4KB of tool definitions, totaling 6544 prompt tokens. With the same
# persistent llama_context reused from a warm-up turn, libllama then fails to
# allocate a 512-token KV slot for the prefill of this shape, returning:
#
#   init_batch: failed to prepare attention ubatches
#   decode: failed to find a memory slot for batch of size 512
#
# All ASCII; no quote/backslash escaping needed when inlining into JSON.

SYS="You are a music librarian for a digital music store with access to the catalog and purchase history via knowledge tools. Start by calling graph_schema to discover entity types and relationships, then use entity_lookup, traverse, search, or sql as needed. When presenting results: format durations as M:SS, include composer credits, use tables for lists, and suggest related music when relevant."

# 5 OpenAI-style tool definitions (~matches the user'\''s 3.4KB tools blob).
TOOLS='[
 {"type":"function","function":{"name":"knowledge_search","description":"Semantic search over the music store catalog.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Descriptive search query"},"k":{"type":"number","description":"Number of results"}},"required":["query"]}}},
 {"type":"function","function":{"name":"knowledge_traverse","description":"Traverse relationships around an entity.","parameters":{"type":"object","properties":{"entityName":{"type":"string"},"entityId":{"type":"string"},"depth":{"type":"number"}}}}},
 {"type":"function","function":{"name":"knowledge_entity_lookup","description":"Find entities by name, ID, or type.","parameters":{"type":"object","properties":{"id":{"type":"string"},"name":{"type":"string"},"type":{"type":"string"},"limit":{"type":"number"},"offset":{"type":"number"}}}}},
 {"type":"function","function":{"name":"knowledge_graph_schema","description":"Get the schema of the music store graph.","parameters":{"type":"object","properties":{"includeExamples":{"type":"boolean"}}}}},
 {"type":"function","function":{"name":"knowledge_sql","description":"Run a readonly SELECT against the music store database.","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"number"}},"required":["query"]}}}
]'

# Tool-result chunk: a single long-ish payload of catalog data, repeated to
# blow each tool message up to ~1.5KB so the full prompt lands near 6K tokens.
trchunk='[1] (score: 0.668) track_id 170, track_name A Statistic, composer , album_id 18, album_title Body Count, artist_id 13, artist_name Body Count, genre_id 4, genre_name Alternative Punk, A Statistic by Body Count on album Body Count duration 0:06. '
trbig=""
for _ in $(seq 1 24); do trbig="$trbig$trchunk"; done

# Warm-up turn — small request that establishes a session with the same
# system+tools prefix turn 2 will share. Persistent-session prefix reuse
# means turn 2 will `seq_rm` to the common length and decode the new tail.
B1="$(mktemp)"
{
    printf '{"messages":[{"role":"system","content":"%s"},{"role":"user","content":"Reply with: ok"}],'
    printf '"tools":'
    printf '%s' "$TOOLS"
    printf ',"max_tokens":8,"temperature":0}'
} > "$B1"
# Strip newlines from system prompt content (JSON-safe).
SYS_FILE="$(mktemp)"; printf '%s' "$SYS" | tr -d '\n' > "$SYS_FILE"
SYS_ESC="$(cat "$SYS_FILE")"; rm -f "$SYS_FILE"
{
    printf '{"messages":[{"role":"system","content":"%s"},{"role":"user","content":"Reply with: ok"}],' "$SYS_ESC"
    printf '"tools":'
    printf '%s' "$TOOLS"
    printf ',"max_tokens":8,"temperature":0}'
} > "$B1"
REQ1="$(curl -fs --max-time 120 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary @"$B1")"
REQ1_RC=$?
rm -f "$B1"
if [ "$REQ1_RC" -eq 0 ] && echo "$REQ1" | grep -q '"finish_reason"'; then
    ok "warm-up turn: tools+system establishes the session"
else
    bad "warm-up turn failed" "rc=$REQ1_RC body=$(echo "$REQ1" | head -c 240)"
fi

# Failure-shaped turn — system+tools (shared prefix) + a user message + an
# assistant message with 3 parallel tool_calls + 3 tool messages each with
# ~1.5KB of result text + a fresh user follow-up. This is the same shape
# the production agent loop produces when the model fans out tool calls.
# Built with python3 to dodge bash's printf-collapses-backslash-quotes trap.
B2="$(mktemp)"
SYS_ESC="$SYS" TR_BIG="$trbig" TOOLS_JSON="$TOOLS" python3 - "$B2" <<'PY'
import json, os, sys
sys_prompt = os.environ['SYS_ESC']
trbig      = os.environ['TR_BIG']
tools      = json.loads(os.environ['TOOLS_JSON'])
body = {
  "messages": [
    {"role": "system", "content": sys_prompt},
    {"role": "user",   "content": "Which artists have tracks in both Rock and Jazz?"},
    {"role": "assistant", "content": None, "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"name": "knowledge_graph_schema",  "arguments": json.dumps({"includeExamples": True})}},
      {"id": "call_2", "type": "function", "function": {"name": "knowledge_entity_lookup", "arguments": json.dumps({"type": "Album", "limit": 20})}},
      {"id": "call_3", "type": "function", "function": {"name": "knowledge_search",        "arguments": json.dumps({"query": "artists with tracks in both Rock and Jazz genres"})}},
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": trbig},
    {"role": "tool", "tool_call_id": "call_2", "content": trbig},
    {"role": "tool", "tool_call_id": "call_3", "content": trbig},
    {"role": "user", "content": "which are the best selling albums?"},
  ],
  "tools":       tools,
  "max_tokens":  64,
  "temperature": 0,
  "stream":      True,
}
with open(sys.argv[1], 'w') as f:
    json.dump(body, f)
PY
[ -n "$SAVED_LOG" ] && cp "$B2" "${SAVED_LOG%.log}.b2.json"
REQ2="$(curl -fs --max-time 180 -N "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary @"$B2")"
REQ2_RC=$?
rm -f "$B2"

if [ "$REQ2_RC" -eq 0 ] && echo "$REQ2" | grep -q 'chat.completion.chunk' && echo "$REQ2" | grep -q '\[DONE\]'; then
    ok "turn 2: long prompt streams cleanly across the persistent session"
else
    bad "turn 2: long prompt failed (the bug)" "curl_rc=$REQ2_RC; body=$(echo "$REQ2" | head -c 400)"
fi

# Server log must NOT contain the SessionSyncFailed cascade. If the retry
# path (Fix #2) kicks in it logs "[llama-cache] sync failed ... retrying
# cold"; that's the safety net firing on a transient failure, which we
# also flag as a soft regression so the SWA fix doesn't silently regress
# into "we just rely on the retry every turn".
if grep -q 'SessionSyncFailed' "$LOG"; then
    bad "no SessionSyncFailed in server log" "found in $LOG"
else
    ok "no SessionSyncFailed in server log"
fi
if grep -q 'sync failed.*retrying cold' "$LOG"; then
    bad "no sync-retry fallback fired (Fix #1 should prevent the failure outright)" \
        "$(grep -m3 'sync failed' "$LOG")"
else
    ok "no sync-retry fallback needed on the happy path"
fi

echo ""
echo -e "  ${GREEN}$PASS passed${NC}, $([ "$FAIL" -gt 0 ] && echo -e "${RED}$FAIL failed${NC}" || echo "0 failed")"
[ "$FAIL" -eq 0 ]
