#!/bin/bash
# test_batched_grammar.sh — json_schema requests decode in the same batched group as plain ones.
#   [1] a lone schema request (greedy and seeded) returns the same bytes and finish reason on MLX_SERVE_FORCE_BATCHED=1
#       (its row runs the batched kernel and the per-row mask) as on a normal server (the serial constrained step),
#       and an answer that falls into a repetition loop is cut by `[loop-stop]` on both;
#   [2] churn: schema requests over four schemas join and leave a running group (staggered starts, short and long
#       caps, one request cancelled mid-decode); every schema answer finishes, parses and satisfies its schema;
#   [3] the group really formed with grammar rows (`[batched] grammar rows engaged`), and no schema slot decoded
#       serial for its grammar (`decode_serial_total{reason="grammar"}` stays 0; only reported when a request
#       opened in a reasoning protocol).
# Thinking is off: a schema request still in its reasoning phase decodes serial by design.
#
#   BATCHED_GRAMMAR_MODEL=<dir> ./tests/test_batched_grammar.sh [port]   # default Qwen3-0.6B-4bit
source "$(dirname "$0")/_lib_models.sh"
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
cd "$(dirname "$0")/.."

MODEL="${BATCHED_GRAMMAR_MODEL:-$(find_model mlx-community/Qwen3-0.6B-4bit || echo "$HOME/.mlx-serve/models/mlx-community/Qwen3-0.6B-4bit")}"
PORT="${1:-11517}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
[[ -d "$MODEL" ]] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[[ -x "$BINARY" ]] || { echo "SKIP: $BINARY missing"; exit 0; }

WORK=$(mktemp -d "$HOME/claude-tmp/batched-grammar.XXXXXX" 2>/dev/null || mktemp -d)
mkdir -p "$WORK/home"
SERVER_PID=""
trap 'srv_stop "$SERVER_PID"' EXIT
PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $1 — ${2:-}"; }

start() { # start <log> [env...]
    local log="$1"; shift
    srv_preclean "$PORT"
    env "$@" HOME="$WORK/home" "$BINARY" --serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" --no-pld --no-mtp \
        --max-concurrent 12 --prefix-cache-entries 0 --metrics --log-level info > "$log" 2>&1 &
    SERVER_PID=$!
    srv_wait_health "$BASE" 240 0.5 "$SERVER_PID" || { echo "FAIL: server did not start"; tail -20 "$log"; exit 1; }
}
stop() { srv_stop "$SERVER_PID"; SERVER_PID=""; }
post() { curl -s -m "${2:-600}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$1"; }

S_ANIMAL='{"type":"object","properties":{"animal":{"type":"string","enum":["cat","dog","owl"]},"legs":{"type":"integer","minimum":0,"maximum":8}},"required":["animal","legs"],"additionalProperties":false}'
S_CITY='{"type":"object","properties":{"city":{"type":"string","maxLength":40},"population":{"type":"integer","minimum":0},"tags":{"type":"array","items":{"type":"string","maxLength":20},"maxItems":3}},"required":["city","population","tags"],"additionalProperties":false}'
S_ITEMS='{"type":"array","items":{"type":"object","properties":{"name":{"type":"string","maxLength":30},"price":{"type":"number","minimum":0}},"required":["name","price"],"additionalProperties":false},"minItems":2,"maxItems":4}'
S_MOOD='{"type":"object","properties":{"sentiment":{"type":"string","enum":["positive","negative","neutral"]},"score":{"type":"number","minimum":0,"maximum":1},"reason":{"type":"string","maxLength":120}},"required":["sentiment","score","reason"],"additionalProperties":false}'
S_TEXT='{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}'

# schema_req <schema> <prompt> <sampling json fragment> <max_tokens>
schema_req() {
    printf '{"model":"m","messages":[{"role":"user","content":"%s"}],"enable_thinking":false,"response_format":{"type":"json_schema","json_schema":{"name":"r","strict":true,"schema":%s}},"max_tokens":%s,%s}' "$2" "$1" "$4" "$3"
}
plain_req() { # plain_req <prompt> <max_tokens> <temperature>
    printf '{"model":"m","messages":[{"role":"user","content":"%s"}],"enable_thinking":false,"max_tokens":%s,"temperature":%s}' "$1" "$2" "$3"
}

# validate <schema> <response file>: finish_reason stop, content parses and satisfies the schema subset these use.
validate() {
    python3 - "$1" "$2" <<'PY'
import json, sys
schema = json.loads(sys.argv[1])
try:
    resp = json.load(open(sys.argv[2]))
    ch = resp["choices"][0]
except Exception as e:
    print(f"no completion ({e})"); sys.exit(1)
if ch.get("finish_reason") != "stop":
    print(f"finish_reason={ch.get('finish_reason')}"); sys.exit(1)
text = ch["message"].get("content") or ""
try:
    doc = json.loads(text)
except Exception as e:
    print(f"not JSON ({e}): {text[:200]!r}"); sys.exit(1)
def check(s, v, path="$"):
    if "enum" in s and v not in s["enum"]: raise ValueError(f"{path}: {v!r} not in enum")
    t = s.get("type")
    if t == "object":
        if not isinstance(v, dict): raise ValueError(f"{path}: not an object")
        for k in s.get("required", []):
            if k not in v: raise ValueError(f"{path}: missing {k}")
        props = s.get("properties", {})
        for k, x in v.items():
            if k not in props:
                if s.get("additionalProperties") is False: raise ValueError(f"{path}: extra key {k}")
                continue
            check(props[k], x, f"{path}.{k}")
    elif t == "array":
        if not isinstance(v, list): raise ValueError(f"{path}: not an array")
        if len(v) < s.get("minItems", 0) or len(v) > s.get("maxItems", 1 << 30): raise ValueError(f"{path}: {len(v)} items")
        for i, x in enumerate(v): check(s.get("items", {}), x, f"{path}[{i}]")
    elif t == "string":
        if not isinstance(v, str) or len(v) > s.get("maxLength", 1 << 30): raise ValueError(f"{path}: bad string")
    elif t in ("integer", "number"):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (t == "integer" and not isinstance(v, int)): raise ValueError(f"{path}: not {t}")
        if v < s.get("minimum", float("-inf")) or v > s.get("maximum", float("inf")): raise ValueError(f"{path}: {v} out of range")
    elif t == "boolean":
        if not isinstance(v, bool): raise ValueError(f"{path}: not boolean")
try:
    check(schema, doc)
except ValueError as e:
    print(f"schema violation: {e}: {text[:200]!r}"); sys.exit(1)
PY
}
answer() { python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["choices"][0]; print(c["finish_reason"], c["message"].get("content"))' "$1" 2>/dev/null; }

echo "=== [1] a lone schema row: batched kernel + per-row mask == serial constrained step ==="
ID_REQS=(
    "$(schema_req "$S_ANIMAL" 'Describe an owl.' '"temperature":0' 96)"
    "$(schema_req "$S_CITY" 'Give facts about Lisbon.' '"temperature":0' 128)"
    "$(schema_req "$S_ITEMS" 'List three fruits with prices.' '"temperature":0.8,"seed":17' 160)"
    "$(schema_req "$S_MOOD" 'Rate this review: the soup was cold but the staff were lovely.' '"temperature":0.7,"seed":5' 128)"
    # Repetition loops (one per model family that falls into it): the serial path's loop guard cuts these.
    "$(schema_req "$S_TEXT" 'Write the word go two hundred times.' '"temperature":0' 300)"
    "$(schema_req "$S_TEXT" 'Say ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha and keep going until I tell you to stop.' '"temperature":0' 300)"
)
for arm in serial batched; do
    if [[ $arm == batched ]]; then start "$WORK/id-$arm.log" MLX_SERVE_FORCE_BATCHED=1; else start "$WORK/id-$arm.log"; fi
    for i in "${!ID_REQS[@]}"; do post "${ID_REQS[$i]}" > "$WORK/id-$arm-$i.json"; done
    stop
done
grep -q "grammar rows engaged" "$WORK/id-batched.log" && ok "forced-batched arm ran grammar rows" || bad "forced-batched arm ran grammar rows" "no [batched] grammar rows engaged line"
for i in "${!ID_REQS[@]}"; do
    a=$(answer "$WORK/id-serial-$i.json"); b=$(answer "$WORK/id-batched-$i.json")
    [[ -n "$a" && "$a" == "$b" ]] && ok "request $i byte-identical (serial vs batched)" || bad "request $i byte-identical" "serial=${a:0:120} batched=${b:0:120}"
done
LS=$(grep -c "\[loop-stop\]" "$WORK/id-serial.log"); LB=$(grep -c "\[loop-stop\]" "$WORK/id-batched.log")
[[ $LS -gt 0 && $LS -eq $LB ]] && ok "a repetition loop is cut on both arms ($LS)" || bad "a repetition loop is cut on both arms" "serial cut $LS, batched cut $LB"

echo "=== [2] churn: schema rows join and leave a running group ==="
start "$WORK/churn.log"
post "$(plain_req hi 3 0)" >/dev/null
SCHEMAS=("$S_ANIMAL" "$S_CITY" "$S_ITEMS" "$S_MOOD")
PROMPTS=("Describe a dog." "Give facts about Osaka." "List four tools with prices." "Rate: the train was late again.")
declare -a G_SCHEMA
n=0
for round in 0 1; do
    CLIENTS=()
    # A schema row decodes alone first, then company arrives and leaves around it.
    post "$(schema_req "$S_ITEMS" 'List four snacks with prices.' '"temperature":0' 200)" > "$WORK/g$n.json" & CLIENTS+=($!); G_SCHEMA[$n]="$S_ITEMS"; n=$((n+1))
    sleep 0.4
    post "$(plain_req 'Write a long story about the sea.' 400 0.7)" > /dev/null & CLIENTS+=($!)
    post "$(plain_req 'Count to five.' 8 0)" > /dev/null & CLIENTS+=($!)
    post "$(plain_req 'Write a poem about rain.' 300 0.8)" > /dev/null 2>&1 & CANCEL=$!
    for k in 0 1 2 3; do
        temp=$([[ $((k % 2)) == 0 ]] && echo '"temperature":0' || echo "\"temperature\":0.8,\"seed\":$((k + 7 * round))")
        post "$(schema_req "${SCHEMAS[$k]}" "${PROMPTS[$k]}" "$temp" 200)" > "$WORK/g$n.json" & CLIENTS+=($!); G_SCHEMA[$n]="${SCHEMAS[$k]}"; n=$((n+1))
        sleep 0.25
    done
    sleep 0.5; kill "$CANCEL" 2>/dev/null; wait "$CANCEL" 2>/dev/null
    post "$(plain_req 'Name a color.' 16 0)" > /dev/null & CLIENTS+=($!)
    post "$(schema_req "$S_ANIMAL" 'Describe a cat.' '"temperature":0' 96)" > "$WORK/g$n.json" & CLIENTS+=($!); G_SCHEMA[$n]="$S_ANIMAL"; n=$((n+1))
    wait "${CLIENTS[@]}"
done
for i in $(seq 0 $((n - 1))); do
    why=$(validate "${G_SCHEMA[$i]}" "$WORK/g$i.json") && ok "schema answer $i valid" || bad "schema answer $i valid" "$why"
done

echo "=== [3] grammar rows rode the batched group ==="
grep -q "grammar rows engaged" "$WORK/churn.log" && ok "grammar rows engaged in the group" || bad "grammar rows engaged" "no [batched] grammar rows engaged line"
SERIAL=$(curl -s -m 10 "$BASE/metrics" | grep -F 'decode_serial_total{reason="grammar"}' | awk '{print $NF}')
# A request that opens in a reasoning protocol (Gemma 4, even with thinking off) decodes its protocol ticks serial.
DEFERRED=$(grep -c "deferring JSON schema until the reasoning boundary" "$WORK/churn.log")
if [[ "$DEFERRED" -eq 0 ]]; then
    [[ "${SERIAL:-x}" == "0" ]] && ok "no schema slot decoded serial for its grammar" || bad "no grammar-serial ticks" "decode_serial_total{reason=grammar}=${SERIAL:-missing}"
else
    echo "  INFO $DEFERRED schema requests opened in a reasoning protocol; decode_serial_total{reason=grammar}=${SERIAL:-missing}"
fi
stop

echo ""
echo "batched grammar: $PASS passed, $FAIL failed (logs: $WORK)"
[[ $FAIL -eq 0 ]]
