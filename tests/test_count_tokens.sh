#!/bin/bash
# POST /v1/messages/count_tokens — the Anthropic token-counting endpoint.
#
# The point of this script is that "200 OK" proves nothing here: a handler
# that returns a hardcoded zero, or one that counts a prompt nobody would
# ever generate from, passes a status check. So every assertion below is
# about the NUMBER:
#
#   • it is a positive integer;
#   • more text in the body means a bigger count (a constant cannot pass);
#   • a system prompt raises it, and tools raise it (the render really is the
#     /v1/messages render, not a naive strlen of the user turn);
#   • and — the real invariant — it EQUALS the `usage.input_tokens` that an
#     actual POST /v1/messages reports for the identical body. That is the
#     whole contract: count_tokens answers for the prompt the real request
#     would build. Nothing else is worth asserting.
#
# Usage: ./tests/test_count_tokens.sh [port]
# Requires a running mlx-serve server with a loaded text model.

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ok()   { TOTAL=$((TOTAL+1)); PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
bad()  { TOTAL=$((TOTAL+1)); FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; [ -n "$2" ] && echo "    $2"; }

count_of() {  # body -> input_tokens (empty on any non-count response)
    curl -s "$BASE/v1/messages/count_tokens" \
        -H 'content-type: application/json' \
        -H 'anthropic-version: 2023-06-01' \
        -d "$1" | grep -o '"input_tokens":[0-9]*' | head -1 | cut -d: -f2
}

status_of() {
    curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/messages/count_tokens" \
        -H 'content-type: application/json' -d "$1"
}

echo "=== POST /v1/messages/count_tokens ($BASE) ==="

MODEL=$(curl -s "$BASE/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "$MODEL" ]; then
    echo "SKIP: no model listed at $BASE/v1/models"
    exit 0
fi
echo "model: $MODEL"

SHORT="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"
LONG="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi. And now a great deal more text, deliberately long enough that no tokenizer on earth could render it in the same number of tokens as the word Hi, because a counter that ignores its input would otherwise pass this test.\"}]}"
SYS="{\"model\":\"$MODEL\",\"system\":\"You are a terse assistant that always answers in exactly one word.\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"
TOOLS="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"tools\":[{\"name\":\"get_weather\",\"description\":\"Look up the current weather for a city\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\",\"description\":\"City name\"}},\"required\":[\"city\"]}}]}"

# ── 1. a positive integer, not zero and not absent
N_SHORT=$(count_of "$SHORT")
if [ -n "$N_SHORT" ] && [ "$N_SHORT" -gt 0 ] 2>/dev/null; then
    ok "short body counts a positive integer ($N_SHORT)"
else
    bad "short body counts a positive integer" "got: '${N_SHORT:-<none>}'"
fi

# ── 2. more text ⇒ a bigger count. A hardcoded number dies here.
N_LONG=$(count_of "$LONG")
if [ -n "$N_LONG" ] && [ -n "$N_SHORT" ] && [ "$N_LONG" -gt "$N_SHORT" ] 2>/dev/null; then
    ok "more message text raises the count ($N_SHORT -> $N_LONG)"
else
    bad "more message text raises the count" "short=$N_SHORT long=$N_LONG"
fi

# ── 3. a system prompt is part of the prompt
N_SYS=$(count_of "$SYS")
if [ -n "$N_SYS" ] && [ "$N_SYS" -gt "$N_SHORT" ] 2>/dev/null; then
    ok "a system prompt raises the count ($N_SHORT -> $N_SYS)"
else
    bad "a system prompt raises the count" "plain=$N_SHORT system=$N_SYS"
fi

# ── 4. tools are serialized into the prompt, so they cost tokens
N_TOOLS=$(count_of "$TOOLS")
if [ -n "$N_TOOLS" ] && [ "$N_TOOLS" -gt "$N_SHORT" ] 2>/dev/null; then
    ok "declaring a tool raises the count ($N_SHORT -> $N_TOOLS)"
else
    bad "declaring a tool raises the count" "plain=$N_SHORT tools=$N_TOOLS"
fi

# ── 5. THE invariant: the count is what /v1/messages itself bills.
#      Run for each shape above — a counter can agree on a bare turn and
#      still diverge the moment tools or a system block enter the render.
for pair in "plain:$SHORT:$N_SHORT" "system:$SYS:$N_SYS" "tools:$TOOLS:$N_TOOLS"; do
    label=${pair%%:*}
    rest=${pair#*:}
    body=${rest%:*}
    counted=${rest##*:}
    gen_body=$(printf '%s' "$body" | sed 's/}$/,"max_tokens":1}/')
    actual=$(curl -s "$BASE/v1/messages" \
        -H 'content-type: application/json' \
        -H 'anthropic-version: 2023-06-01' \
        -d "$gen_body" | grep -o '"input_tokens":[0-9]*' | head -1 | cut -d: -f2)
    if [ -n "$actual" ] && [ "$actual" = "$counted" ]; then
        ok "$label: count_tokens == /v1/messages usage.input_tokens ($actual)"
    else
        bad "$label: count_tokens == /v1/messages usage.input_tokens" \
            "count_tokens=$counted  /v1/messages=${actual:-<none>}"
    fi
done

# ── 6. named 400s, not crashes
S=$(status_of "{\"model\":\"$MODEL\"}")
[ "$S" = "400" ] && ok "missing 'messages' is a 400" || bad "missing 'messages' is a 400" "got $S"

S=$(status_of "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}")
[ "$S" = "400" ] && ok "missing 'model' is a 400" || bad "missing 'model' is a 400" "got $S"

S=$(status_of "not json at all")
[ "$S" = "400" ] && ok "invalid JSON is a 400" || bad "invalid JSON is a 400" "got $S"

ERR=$(curl -s "$BASE/v1/messages/count_tokens" -H 'content-type: application/json' -d "{\"model\":\"$MODEL\"}")
echo "$ERR" | grep -q '"type":"error"' && echo "$ERR" | grep -q 'invalid_request_error' \
    && ok "the 400 uses the Anthropic error shape" \
    || bad "the 400 uses the Anthropic error shape" "$ERR"

# ── 7. the server is still alive and serving after all of that
S=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")
[ "$S" = "200" ] && ok "server healthy after count_tokens traffic" || bad "server healthy after count_tokens traffic" "got $S"

echo
echo "  $PASS/$TOTAL passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
