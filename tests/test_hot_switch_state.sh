#!/bin/bash
# test_hot_switch_state.sh — every model answers with ITS OWN tokenizer, template and grammar table. Several models
# interleave on one server (A, B, C, then A again: a singleton built by A leaks into B, and A's second visit catches a
# table rebuilt for B).
#
#   ./tests/test_hot_switch_state.sh [model_dir_root] [port]   # HOT_SWITCH_MODELS=<csv of ids>
#
# Per model per round: /tokenize returns a stable token count for that model across switches, a plain turn answers
# with no markup leak, `json_object` yields parseable JSON, and the log shows a grammar build for that model. A
# failure is bad OUTPUT, not a crash (the per-model grammar table story in docs/gotchas/server-http.md).

set -u
source "$(dirname "$0")/lib/server.sh"

ROOT="${1:-$HOME/.mlx-serve/models}"
PORT="${2:-8134}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; DIM='\033[2m'; NC='\033[0m'
PASS=0; FAIL=0

ok()  { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
bad() { echo -e "  ${RED}FAIL${NC} $1"; shift; for l in "$@"; do echo "        $l"; done; FAIL=$((FAIL+1)); }

[ -d "$ROOT" ] || { echo "SKIP: no model root at $ROOT"; exit 0; }
[ -x ./zig-out/bin/mlx-serve ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

LOG=$(mktemp /tmp/hot_switch.XXXXXX)
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 1
./zig-out/bin/mlx-serve serve --port "$PORT" --host 127.0.0.1 --model-dir "$ROOT" \
    --log-level debug > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; rm -f "$LOG" "${TOKCOUNT_FILE:-}"; }
trap cleanup EXIT

srv_wait_health "$BASE" 90 1 || { echo "FAIL: server did not start"; exit 1; }

# Pick the smallest text models the box actually has — this test is about state
# leaking BETWEEN models, so the cheapest two or three that differ in family are
# worth more than one big one.
if [ -n "${HOT_SWITCH_MODELS:-}" ]; then
    IFS=',' read -r -a MODELS <<< "$HOT_SWITCH_MODELS"
else
    # macOS ships bash 3.2 — no mapfile.
    MODELS=()
    while IFS= read -r line; do [ -n "$line" ] && MODELS+=("$line"); done < <(curl -sf "$BASE/v1/models" | python3 -c "
import json, sys
d = json.load(sys.stdin)['data']
def ok(m):
    caps = m.get('capabilities') or []
    meta = m.get('meta') or {}
    # Local packs only: a provider or LAN row is someone else's server.
    local = 'provider' not in m and m.get('state') != 'remote'
    # An embedder on a decoder arch advertises chat too, and only babbles.
    return local and 'chat' in caps and 'embeddings' not in caps and (meta.get('bytes_on_disk') or m.get('bytes_on_disk') or 0) < 12e9
picked = [m['id'] for m in sorted(d, key=lambda m: m.get('bytes_on_disk') or 0) if ok(m)]
print('\n'.join(picked[:3]))
" 2>/dev/null)
fi
set +u
[ "${#MODELS[@]:-0}" -ge 2 ] || { echo "SKIP: need >=2 small chat models under $ROOT (found ${#MODELS[@]})"; exit 0; }
echo "models: ${MODELS[*]}"

# A string that pins the tokenizer: mixed digit runs (the digit-GROUPING class,
# per-model) plus multi-byte text plus punctuation.
PROBE_TEXT='Order 1234567 shipped 89 units — 北京, café, "quoted"'

TOKCOUNT_FILE=$(mktemp /tmp/hot_switch_tok.XXXXXX)

# Request bodies are built by python into a FILE and posted with --data-binary.
# Inlining a multi-line `python3 -c` inside $( ) lets the shell word-split the
# program, which silently produced invalid python (and an empty request).
mkreq() { # model json_fragment_file_out kind
    python3 - "$1" "$2" "$3" <<'PY'
import json, sys
model, out, kind = sys.argv[1], sys.argv[2], sys.argv[3]
if kind == "tokenize":
    body = {"model": model,
            "content": 'Order 1234567 shipped 89 units \u2014 \u5317\u4eac, caf\u00e9, "quoted"'}
elif kind == "chat":
    body = {"model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: apple"}],
            "max_tokens": 24, "temperature": 0}
else:
    body = {"model": model,
            "messages": [{"role": "user",
                          "content": "Give a JSON object with keys name and color for an apple."}],
            "max_tokens": 120, "temperature": 0,
            "response_format": {"type": "json_object"}}
open(out, "w").write(json.dumps(body))
PY
}

probe_model() { # id round
    local id="$1" round="$2"
    local body; body=$(mktemp /tmp/hot_switch_req.XXXXXX)

    # ── tokenizer ──
    mkreq "$id" "$body" tokenize
    local n
    n=$(curl -sf "$BASE/tokenize" -H 'Content-Type: application/json' --data-binary @"$body" 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('tokens') or []))" 2>/dev/null)
    n="${n:-0}"
    if [ "$n" -gt 0 ]; then
        local prev; prev=$(grep -F "$id=" "$TOKCOUNT_FILE" 2>/dev/null | head -1 | sed "s/.*=//")
        if [ -z "$prev" ]; then
            echo "$id=$n" >> "$TOKCOUNT_FILE"
            ok "[$id r$round] tokenizer: $n tokens"
        elif [ "$prev" = "$n" ]; then
            ok "[$id r$round] tokenizer count stable across the switch ($n)"
        else
            bad "[$id r$round] tokenizer count changed after a switch" "was $prev, now $n — a foreign tokenizer served this model"
        fi
    else
        echo -e "  ${DIM}skip [$id r$round] /tokenize unavailable${NC}"
    fi

    # ── template ──
    mkreq "$id" "$body" chat
    local reply
    reply=$(curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @"$body" 2>/dev/null \
        | python3 -c "import json,sys; print((json.load(sys.stdin)['choices'][0]['message'].get('content') or '').strip())" 2>/dev/null)
    if [ -n "$reply" ]; then
        case "$reply" in
            *"<|"*|*"<tool_call"*|*"</think"*|*"<|channel"*)
                bad "[$id r$round] template: markup leaked into content" "$reply" ;;
            *) ok "[$id r$round] template: clean reply (${reply:0:40})" ;;
        esac
    else
        bad "[$id r$round] template: empty reply"
    fi

    # ── grammar ──
    mkreq "$id" "$body" json
    local js
    js=$(curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @"$body" 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message'].get('content') or '')" 2>/dev/null)
    if [ -n "$js" ] && echo "$js" | python3 -c "import json,sys; json.loads(sys.stdin.read())" 2>/dev/null; then
        ok "[$id r$round] grammar: json_object output parses"
    else
        bad "[$id r$round] grammar: json_object output is not JSON" "${js:0:160}"
    fi
    rm -f "$body"
}

# Interleave: A B (C) A — a singleton built by A leaks into B, and A's second
# visit is where a table rebuilt for B is caught.
ROUND=1
for id in "${MODELS[@]}"; do probe_model "$id" "$ROUND"; done
ROUND=2
for id in "${MODELS[@]}"; do probe_model "$id" "$ROUND"; done

# The grammar table is per-model, so each model that answered a constrained
# request must have built its OWN. One build line total = the singleton class.
BUILDS=$(grep -c "\[grammar\]" "$LOG" 2>/dev/null); BUILDS="${BUILDS:-0}"
if [ "$BUILDS" -ge "${#MODELS[@]}" ]; then
    ok "grammar tables are per-model ($BUILDS build/enforce lines for ${#MODELS[@]} models)"
else
    bad "grammar tables are per-model" "only $BUILDS [grammar] lines for ${#MODELS[@]} models — a shared table would log once"
fi

echo
echo "hot-switch state: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
