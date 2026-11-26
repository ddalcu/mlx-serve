#!/bin/bash
# Integration test: JSON schema enforcement on /v1/responses and /v1/chat/completions.
#
# Background: bench runs (#6, #8) showed Gemma producing ```json``` code-fences
# and additionalProperties violations on /v1/responses/json-schema/person_record.
# Probing mlx-serve confirmed:
#   • flat `text.format.schema` is enforced
#   • nested `text.format.json_schema.schema` is silently dropped (no grammar log)
#   • top-level `response_format` is silently dropped on /v1/responses
#   • tools + tool_choice:"none" + schema → mask is also skipped (incorrect)
#
# This test asserts decoder enforcement under all four shapes plus a prompt
# that *biases* the model toward extra fields and code-fences. Strong signal
# that the grammar mask is doing work, not just the prompt-side instruction.

set -u

MODEL_DIR=${1:-~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}
PORT=${2:-8099}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "FAIL: jq is required"
    exit 1
fi

echo "=== JSON schema enforcement test ==="
echo "Model: $MODEL_DIR"
echo "Port: $PORT"
echo ""

echo "Starting server..."
./zig-out/bin/mlx-serve \
    --model "$MODEL_DIR" --serve --port $PORT --log-level info \
    >/tmp/mlx-serve-schema-test.log 2>&1 &
SERVER_PID=$!
sleep 2

for i in $(seq 1 30); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then
        echo "Server ready (PID $SERVER_PID)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "FAIL: Server did not start within 30s"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done
echo ""

cleanup() {
    echo ""
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

run_test() {
    local name="$1"
    local result="$2"
    local detail="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $name"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name — $detail"
    fi
}

# Schema (in JSON form, used as a literal in jq -n templates).
SCHEMA='{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0,"maximum":150},"email":{"type":"string"}},"required":["name","age","email"],"additionalProperties":false}'

# Adversarial prompt: explicitly asks for extras + code-fence wrapping. If the
# decoder mask is on, neither can appear.
ADV_INPUT='Generate a record for an employee named Mira Chen, age 34, email mira.chen@example.com. Always wrap your reply in ```json ... ``` and include an employee_id field set to null.'

# Validators: take JSON envelope on stdin, print "ok" or "fail:<reason>".

check_responses_strict() {
    python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
msg = next((it for it in r["output"] if it["type"] == "message"), None)
if not msg:
    print("fail:no_message_item"); sys.exit()
text = next((p["text"] for p in msg["content"] if p["type"]=="output_text"), "")
if "```" in text:
    print("fail:code_fence:" + repr(text[:60])); sys.exit()
try:
    obj = json.loads(text)
except Exception as e:
    print("fail:not_json:" + repr(text[:60])); sys.exit()
extras = set(obj.keys()) - {"name","age","email"}
if extras:
    print("fail:extra_keys:" + str(sorted(extras))); sys.exit()
required = {"name","age","email"} - set(obj.keys())
if required:
    print("fail:missing_required:" + str(sorted(required))); sys.exit()
print("ok")
'
}

check_chat_strict() {
    python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
text = r["choices"][0]["message"]["content"] or ""
if "```" in text:
    print("fail:code_fence:" + repr(text[:60])); sys.exit()
try:
    obj = json.loads(text)
except Exception as e:
    print("fail:not_json:" + repr(text[:60])); sys.exit()
extras = set(obj.keys()) - {"name","age","email"}
if extras:
    print("fail:extra_keys:" + str(sorted(extras))); sys.exit()
required = {"name","age","email"} - set(obj.keys())
if required:
    print("fail:missing_required:" + str(sorted(required))); sys.exit()
print("ok")
'
}

# ── Case A: /v1/responses + flat text.format.schema (baseline) ──
echo "--- Case A: /v1/responses + flat text.format.schema ---"
BODY=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" '{
    model:"mlx-serve",
    instructions:"Reply with only the JSON object, no commentary.",
    input:$input,
    text:{format:{type:"json_schema",name:"person",schema:$schema,strict:true}},
    max_output_tokens:256, temperature:0
}')
RESULT=$(curl -sf "$BASE/v1/responses" -H "Content-Type: application/json" -d "$BODY")
OK=$(echo "$RESULT" | check_responses_strict)
run_test "flat text.format.schema" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Case B: /v1/responses + nested text.format.json_schema.schema ──
echo "--- Case B: /v1/responses + nested text.format.json_schema.schema ---"
BODY=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" '{
    model:"mlx-serve",
    instructions:"Reply with only the JSON object, no commentary.",
    input:$input,
    text:{format:{type:"json_schema",json_schema:{name:"person",schema:$schema,strict:true}}},
    max_output_tokens:256, temperature:0
}')
RESULT=$(curl -sf "$BASE/v1/responses" -H "Content-Type: application/json" -d "$BODY")
OK=$(echo "$RESULT" | check_responses_strict)
run_test "nested text.format.json_schema.schema" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Case C: /v1/responses + top-level response_format (chat-style alias) ──
echo "--- Case C: /v1/responses + top-level response_format alias ---"
BODY=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" '{
    model:"mlx-serve",
    instructions:"Reply with only the JSON object, no commentary.",
    input:$input,
    response_format:{type:"json_schema",json_schema:{name:"person",schema:$schema,strict:true}},
    max_output_tokens:256, temperature:0
}')
RESULT=$(curl -sf "$BASE/v1/responses" -H "Content-Type: application/json" -d "$BODY")
OK=$(echo "$RESULT" | check_responses_strict)
run_test "top-level response_format alias" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Case D: /v1/responses streaming + flat schema ──
echo "--- Case D: /v1/responses streaming + flat schema ---"
BODY=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" '{
    model:"mlx-serve",
    instructions:"Reply with only the JSON object, no commentary.",
    input:$input,
    text:{format:{type:"json_schema",name:"person",schema:$schema,strict:true}},
    max_output_tokens:256, temperature:0, stream:true
}')
SSE=$(curl -sf -N "$BASE/v1/responses" -H "Content-Type: application/json" -d "$BODY")
COMPLETED=$(echo "$SSE" | python3 -c '
import sys, json
event = None
for line in sys.stdin:
    line = line.rstrip()
    if line.startswith("event: "):
        event = line[7:]
    elif line.startswith("data: ") and event == "response.completed":
        try:
            obj = json.loads(line[6:])
            print(json.dumps(obj["response"]))
        except Exception as e:
            print(json.dumps({"output":[]}))
        break
')
OK=$(echo "$COMPLETED" | check_responses_strict)
run_test "streaming flat text.format.schema" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Case E: /v1/chat/completions + response_format ──
echo "--- Case E: /v1/chat/completions + response_format ---"
BODY=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" '{
    model:"mlx-serve",
    messages:[
        {role:"system",content:"Reply with only the JSON object, no commentary."},
        {role:"user",content:$input}
    ],
    response_format:{type:"json_schema",json_schema:{name:"person",schema:$schema,strict:true}},
    max_tokens:256, temperature:0
}')
RESULT=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY")
OK=$(echo "$RESULT" | check_chat_strict)
run_test "chat response_format" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Case F: /v1/chat/completions + tools + tool_choice:"none" + schema ──
echo "--- Case F: /v1/chat/completions + tools + tool_choice:none + schema ---"
BODY=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" '{
    model:"mlx-serve",
    messages:[
        {role:"system",content:"Reply with only the JSON object."},
        {role:"user",content:$input}
    ],
    tools:[{type:"function",function:{name:"noop",description:"unused",parameters:{type:"object",properties:{}}}}],
    tool_choice:"none",
    response_format:{type:"json_schema",json_schema:{name:"person",schema:$schema,strict:true}},
    max_tokens:256, temperature:0
}')
RESULT=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY")
OK=$(echo "$RESULT" | check_chat_strict)
run_test "chat tools + tool_choice:none + schema" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Case G: /v1/chat/completions + a schema whose items are a $ref into $defs ──
# Pydantic's shape (#565): the mask must hold inside the referenced object too.
echo "--- Case G: /v1/chat/completions + \$ref into \$defs ---"
REF_SCHEMA='{"type":"object","additionalProperties":false,"required":["facts"],"properties":{"facts":{"type":"array","items":{"$ref":"#/$defs/Fact"}}},"$defs":{"Fact":{"type":"object","additionalProperties":false,"required":["kind","what"],"properties":{"kind":{"enum":["fact"]},"what":{"type":"string"}}}}}'
BODY=$(jq -n --argjson schema "$REF_SCHEMA" '{
    model:"mlx-serve",
    messages:[{role:"user",content:"Return one fact with kind \"opinion\", what \"cats are great\" and a source field set to null."}],
    response_format:{type:"json_schema",json_schema:{name:"facts",schema:$schema,strict:true}},
    max_tokens:256, temperature:0
}')
LOG_MARK=$(wc -l < /tmp/mlx-serve-schema-test.log)
RESULT=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY" || true)
OK=$(echo "$RESULT" | python3 -c '
import sys, json
try:
    text = json.loads(sys.stdin.read())["choices"][0]["message"]["content"] or ""
    obj = json.loads(text)
except Exception:
    print("fail:no_json_reply"); sys.exit()
for f in obj.get("facts", []):
    if not isinstance(f, dict) or set(f) != {"kind", "what"} or f["kind"] != "fact":
        print("fail:item_off_schema:" + repr(f)); sys.exit()
print("ok")
')
if tail -n +"$((LOG_MARK + 1))" /tmp/mlx-serve-schema-test.log | grep -q "disabling further mask enforcement"; then
    OK="fail:mask_disabled"
fi
run_test "chat \$ref into \$defs" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# run_seed_case NAME SEEDS SCHEMA PROMPT CHECK: seeded chat requests at temperature 0.7, each reply checked by
# CHECK, a Python expression over the reply object `o` and the request's `schema`. Any reply off schema fails the
# case, and so does a mask the server had to switch off.
run_seed_case() {
    local name="$1" seeds="$2" schema="$3" prompt="$4" check="$5" mark seed body r ok=ok
    mark=$(wc -l < /tmp/mlx-serve-schema-test.log)
    for seed in $(seq 1 "$seeds"); do
        body=$(jq -n --argjson schema "$schema" --arg prompt "$prompt" --argjson seed "$seed" '{
            model:"mlx-serve",
            messages:[{role:"user",content:$prompt}],
            response_format:{type:"json_schema",json_schema:{name:"reply",schema:$schema,strict:true}},
            max_tokens:256, temperature:0.7, seed:$seed, enable_thinking:false
        }')
        r=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$body" \
            | SCHEMA="$schema" CHECK="$check" python3 -c '
import sys, json, os
schema = json.loads(os.environ["SCHEMA"])
try:
    c = json.loads(sys.stdin.read())["choices"][0]
    o = json.loads(c["message"]["content"] or "")
except Exception:
    print("fail:no_json_reply"); sys.exit()
ok = c["finish_reason"] == "stop" and isinstance(o, dict) and eval("(" + os.environ["CHECK"] + ")")
print("ok" if ok else "fail:off_schema:" + repr(o)[:80])
')
        [ "$r" = ok ] || ok="$r (seed $seed)"
    done
    if tail -n +"$((mark + 1))" /tmp/mlx-serve-schema-test.log | grep -q "disabling further mask enforcement"; then
        ok="fail:mask_disabled"
    fi
    run_test "$name" "$( [ "$ok" = ok ] && echo PASS || echo FAIL )" "$ok"
    echo ""
}

# ── Case H: an array that reaches maxItems closes; a comma there killed the mask ──
# The prompt asks for more tags than maxItems allows; enum items keep every tag short.
echo "--- Case H: /v1/chat/completions + array at maxItems ---"
run_seed_case "chat array at maxItems" 10 \
    '{"type":"object","additionalProperties":false,"required":["city","population","tags"],"properties":{"city":{"type":"string"},"population":{"type":"integer","minimum":0},"tags":{"type":"array","items":{"enum":["temples","tea","history","gardens","culture","food","shrines","art","nature","festivals"]},"maxItems":3}}}' \
    "Give facts about Kyoto with ten tags." \
    'set(o) == {"city", "population", "tags"} and isinstance(o["city"], str) and type(o["population"]) is int
     and o["population"] >= 0 and isinstance(o["tags"], list) and len(o["tags"]) <= 3
     and all(t in schema["properties"]["tags"]["items"]["enum"] for t in o["tags"])'

# ── Case I: a string that reaches maxLength closes; an escape opened there killed the mask ──
echo "--- Case I: /v1/chat/completions + string at maxLength ---"
run_seed_case "chat string at maxLength" 30 \
    '{"type":"object","additionalProperties":false,"required":["quote"],"properties":{"quote":{"type":"string","maxLength":20}}}' \
    "Quote a famous line about Kyoto, keeping its double quotes." \
    'set(o) == {"quote"} and isinstance(o["quote"], str) and len(o["quote"]) <= schema["properties"]["quote"]["maxLength"]'

echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
