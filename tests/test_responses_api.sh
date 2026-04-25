#!/bin/bash
# Integration test: /v1/responses (OpenAI Responses API).
#
# Covers MVP non-streaming paths from the implementation plan:
#   1. simple text input
#   2. instructions field (system prompt)
#   4. function tool round-trip
#   6. structured output via text.format.json_schema
#   9. store + GET + DELETE round-trip
#  10. store=false → GET 404
#  11. previous_response_id not found → 404
#
# Streaming cases (3, 8) emit named SSE events.
#
# Usage: ./tests/test_responses_api.sh [model_dir] [port]

MODEL_DIR=${1:-~/.mlx-serve/models/gemma-4-e4b-it-8bit}
PORT=${2:-8099}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

echo "=== /v1/responses Integration Test ==="
echo "Model: $MODEL_DIR"
echo "Port: $PORT"
echo ""

echo "Starting server..."
./zig-out/bin/mlx-serve --model "$MODEL_DIR" --serve --port $PORT --log-level info 2>/tmp/mlx-serve-test-responses.log &
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

# ── Test 1: simple text input ──
echo "--- Test 1: simple text input ---"
RESULT=$(curl -sf "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","input":"Reply with exactly the word OK and nothing else.","max_output_tokens":16,"temperature":0}')

OK=$(echo "$RESULT" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    assert r['object'] == 'response', f'object={r.get(\"object\")}'
    assert r['status'] in ('completed', 'incomplete'), f'status={r.get(\"status\")}'
    output = r.get('output') or []
    assert len(output) >= 1, 'output empty'
    msg = next((it for it in output if it['type'] == 'message'), None)
    assert msg is not None, 'no message item'
    parts = msg.get('content') or []
    assert any(p['type'] == 'output_text' and isinstance(p['text'], str) for p in parts), 'no output_text'
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
run_test "simple text input" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo "  raw: $(echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(json.dumps(r.get('output'))[:120])" 2>/dev/null)"
echo ""

# ── Test 2: instructions ──
echo "--- Test 2: instructions field ---"
RESULT=$(curl -sf "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","instructions":"You always reply in ALL CAPS.","input":"say hi","max_output_tokens":24,"temperature":0}')

TEXT=$(echo "$RESULT" | python3 -c "
import sys, json
r = json.loads(sys.stdin.read())
msg = next((it for it in r['output'] if it['type'] == 'message'), {})
parts = msg.get('content') or []
print(next((p['text'] for p in parts if p['type']=='output_text'), ''))
" 2>/dev/null)
echo "  reply: $TEXT"
# At least one uppercase letter and no all-lowercase reply (some lenience for punctuation)
if echo "$TEXT" | grep -qE '[A-Z]' && ! echo "$TEXT" | grep -qE '^[a-z]+'; then
    run_test "instructions affect reply" "PASS" ""
else
    run_test "instructions affect reply" "FAIL" "expected at least one uppercase letter, got: $TEXT"
fi
echo ""

# ── Test 4: function tool ──
echo "--- Test 4: function tool emits function_call output item ---"
RESULT=$(curl -sf "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"mlx-serve",
    "input":"What is the weather in Paris? Use the get_weather tool.",
    "tools":[{"type":"function","name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
    "tool_choice":"required",
    "max_output_tokens":128,
    "temperature":0
  }')

OK=$(echo "$RESULT" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    fc = next((it for it in r['output'] if it['type'] == 'function_call'), None)
    assert fc is not None, 'no function_call output item'
    assert fc.get('name') == 'get_weather', f'unexpected fn: {fc.get(\"name\")}'
    args = json.loads(fc['arguments'])
    assert isinstance(args.get('city'), str), 'city arg missing'
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
run_test "function tool round-trip" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Test 4b: function tool with json_schema still emits function_call ──
echo "--- Test 4b: function tool with json_schema emits function_call ---"
RESULT=$(curl -sf "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"mlx-serve",
    "input":"What is the weather in Miami? Use the get_weather tool.",
    "tools":[{"type":"function","name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
    "tool_choice":"required",
    "text":{"format":{"type":"json_schema","name":"final_answer","schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}},
    "max_output_tokens":128,
    "temperature":0
  }')

OK_SCHEMA_TOOL=$(echo "$RESULT" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    fc = next((it for it in r['output'] if it['type'] == 'function_call'), None)
    assert fc is not None, 'no function_call output item'
    assert fc.get('name') == 'get_weather', f'unexpected fn: {fc.get(\"name\")}'
    args = json.loads(fc['arguments'])
    assert isinstance(args.get('city'), str), 'city arg missing'
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
run_test "function tool with json_schema" "$( [ "$OK_SCHEMA_TOOL" = ok ] && echo PASS || echo FAIL )" "$OK_SCHEMA_TOOL"
echo ""

# ── Test 4c: optional tools with json_schema still format direct answers ──
echo "--- Test 4c: optional tools with json_schema formats direct answer ---"
RESULT=$(curl -sf "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"mlx-serve",
    "input":"Do not call any tools. Reply with a JSON object whose answer is beach.",
    "tools":[{"type":"function","name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
    "text":{"format":{"type":"json_schema","name":"direct_answer","schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}},
    "max_output_tokens":96,
    "temperature":0
  }')

OK_OPTIONAL_SCHEMA=$(echo "$RESULT" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    fc = next((it for it in r['output'] if it['type'] == 'function_call'), None)
    assert fc is None, 'unexpected function_call output item'
    msg = next((it for it in r['output'] if it['type'] == 'message'), None)
    text = next((p['text'] for p in msg['content'] if p['type']=='output_text'))
    obj = json.loads(text)
    assert isinstance(obj.get('answer'), str), 'answer missing'
    extra = set(obj.keys()) - {'answer'}
    assert not extra, f'extra keys: {extra}'
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
run_test "optional tools with json_schema direct answer" "$( [ "$OK_OPTIONAL_SCHEMA" = ok ] && echo PASS || echo FAIL )" "$OK_OPTIONAL_SCHEMA"
echo ""

# ── Test 5: previous_response_id + function_call_output round-trip ──
echo "--- Test 5: function tool result round-trip ---"
if [ "$OK" = ok ]; then
    RESP_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
    CALL_ID=$(echo "$RESULT" | python3 -c "
import sys, json
r = json.loads(sys.stdin.read())
fc = next((it for it in r['output'] if it['type'] == 'function_call'), {})
print(fc.get('call_id', ''))
" 2>/dev/null)
    ROUNDTRIP=$(curl -sf "$BASE/v1/responses" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\":\"mlx-serve\",
        \"previous_response_id\":\"$RESP_ID\",
        \"input\":[{\"type\":\"function_call_output\",\"call_id\":\"$CALL_ID\",\"output\":\"The weather in Paris is sunny and 21 C.\"}],
        \"max_output_tokens\":64,
        \"temperature\":0
      }")
    OK2=$(echo "$ROUNDTRIP" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    msg = next((it for it in r.get('output', []) if it.get('type') == 'message'), None)
    assert msg is not None, 'no message output item'
    text = next((p.get('text', '') for p in msg.get('content', []) if p.get('type') == 'output_text'), '')
    assert text.strip(), 'empty text'
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
    run_test "function tool result round-trip" "$( [ "$OK2" = ok ] && echo PASS || echo FAIL )" "$OK2"
else
    run_test "function tool result round-trip" "FAIL" "initial function_call failed"
fi
echo ""

# ── Test 6: text.format.json_schema enforces structure ──
echo "--- Test 6: text.format.json_schema strict ---"
RESULT=$(curl -sf "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"mlx-serve",
    "input":"Make up a person record.",
    "text":{"format":{"type":"json_schema","name":"person","schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}}},
    "max_output_tokens":128,
    "temperature":0.7
  }')

OK=$(echo "$RESULT" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    msg = next((it for it in r['output'] if it['type'] == 'message'), None)
    text = next((p['text'] for p in msg['content'] if p['type']=='output_text'))
    obj = json.loads(text)
    assert isinstance(obj.get('name'), str)
    assert isinstance(obj.get('age'), int)
    extra = set(obj.keys()) - {'name','age'}
    assert not extra, f'extra keys: {extra}'
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
run_test "text.format.json_schema strict" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Test 7: structured output default budget avoids truncation ──
echo "--- Test 7: json_schema default max_output_tokens ---"
RESULT=$(curl -sf "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"mlx-serve",
    "input":"Say hello and ask one short cruise discovery question. Set every profile_update field to null.",
    "text":{"format":{"type":"json_schema","name":"cruise_response","schema":{
      "type":"object",
      "properties":{
        "blocks":{"type":"array","items":{"anyOf":[
          {"type":"object","properties":{"type":{"enum":["text"]},"content":{"type":"string"}},"required":["type","content"],"additionalProperties":false},
          {"type":"object","properties":{"type":{"enum":["question"]},"prompt":{"type":"string"},"options":{"type":"array","items":{"type":"string"}},"allowMultiple":{"type":"boolean"}},"required":["type","prompt","options","allowMultiple"],"additionalProperties":false}
        ]}},
        "profile_update":{"type":"object","properties":{
          "vibes":{"anyOf":[{"type":"array","items":{"type":"string"}},{"type":"null"}]},
          "partyType":{"anyOf":[{"type":"array","items":{"type":"string"}},{"type":"null"}]},
          "partySize":{"anyOf":[{"type":"number"},{"type":"null"}]},
          "childAges":{"anyOf":[{"type":"array","items":{"type":"number"}},{"type":"null"}]},
          "travelMonth":{"anyOf":[{"type":"number"},{"type":"null"}]},
          "travelYear":{"anyOf":[{"type":"number"},{"type":"null"}]},
          "durationMin":{"anyOf":[{"type":"number"},{"type":"null"}]},
          "durationMax":{"anyOf":[{"type":"number"},{"type":"null"}]},
          "budgetMin":{"anyOf":[{"type":"number"},{"type":"null"}]},
          "budgetMax":{"anyOf":[{"type":"number"},{"type":"null"}]},
          "departurePort":{"anyOf":[{"type":"string"},{"type":"null"}]},
          "cruiseLine":{"anyOf":[{"type":"string"},{"type":"null"}]},
          "stateOfResidence":{"anyOf":[{"type":"string"},{"type":"null"}]}
        },"required":["vibes","partyType","partySize","childAges","travelMonth","travelYear","durationMin","durationMax","budgetMin","budgetMax","departurePort","cruiseLine","stateOfResidence"],"additionalProperties":false}
      },
      "required":["blocks","profile_update"],
      "additionalProperties":false
    }}},
    "temperature":0
  }')

OK=$(echo "$RESULT" | python3 -c "
import sys, json
try:
    r = json.loads(sys.stdin.read())
    assert r.get('status') == 'completed', f'status={r.get(\"status\")}'
    msg = next((it for it in r['output'] if it['type'] == 'message'), None)
    text = next((p['text'] for p in msg['content'] if p['type']=='output_text'))
    obj = json.loads(text)
    assert isinstance(obj.get('blocks'), list), 'missing blocks'
    assert isinstance(obj.get('profile_update'), dict), 'missing profile_update'
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
run_test "json_schema default max_output_tokens" "$( [ "$OK" = ok ] && echo PASS || echo FAIL )" "$OK"
echo ""

# ── Test 9: store + GET + DELETE ──
echo "--- Test 9: store + GET + DELETE round-trip ---"
RESULT=$(curl -sf "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","input":"Reply with exactly the word OK and nothing else.","max_output_tokens":16,"temperature":0,"store":true}')

RESP_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
echo "  stored id: $RESP_ID"

if [ -z "$RESP_ID" ]; then
    run_test "stored response has id" "FAIL" "no id returned"
else
    # GET
    GET_RESULT=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/v1/responses/$RESP_ID")
    if [ "$GET_RESULT" = "200" ]; then
        run_test "GET stored response" "PASS" ""
    else
        run_test "GET stored response" "FAIL" "got HTTP $GET_RESULT"
    fi

    # DELETE
    DEL_RESULT=$(curl -sf -X DELETE "$BASE/v1/responses/$RESP_ID")
    DELETED=$(echo "$DEL_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('deleted'))" 2>/dev/null)
    if [ "$DELETED" = "True" ]; then
        run_test "DELETE returns deleted=true" "PASS" ""
    else
        run_test "DELETE returns deleted=true" "FAIL" "got $DELETED"
    fi

    # GET after DELETE should be 404
    AFTER_DEL=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/v1/responses/$RESP_ID")
    if [ "$AFTER_DEL" = "404" ]; then
        run_test "GET after DELETE returns 404" "PASS" ""
    else
        run_test "GET after DELETE returns 404" "FAIL" "got HTTP $AFTER_DEL"
    fi
fi
echo ""

# ── Test 10: store=false → GET 404 ──
echo "--- Test 10: store=false skips persistence ---"
RESULT=$(curl -sf "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","input":"Hi","max_output_tokens":8,"temperature":0,"store":false}')

RESP_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
if [ -z "$RESP_ID" ]; then
    run_test "store=false still returns id" "FAIL" "no id"
else
    AFTER_FETCH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/v1/responses/$RESP_ID")
    if [ "$AFTER_FETCH" = "404" ]; then
        run_test "GET non-stored response returns 404" "PASS" ""
    else
        run_test "GET non-stored response returns 404" "FAIL" "got HTTP $AFTER_FETCH"
    fi
fi
echo ""

# ── Test 3: streaming text emits expected events in order ──
echo "--- Test 3: streaming named events ---"
EVENTS=$(curl -sf -N "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","input":"Reply with exactly the word OK and nothing else.","max_output_tokens":16,"temperature":0,"stream":true}' 2>/dev/null)

EVENT_ORDER=$(echo "$EVENTS" | grep -E "^event: " | awk '{print $2}' | tr '\n' ',')
echo "  events: $EVENT_ORDER"
MISSING=""
for required in "response.created" "response.in_progress" "response.output_item.added" "response.output_text.delta" "response.output_text.done" "response.completed"; do
    if ! echo "$EVENT_ORDER" | grep -q "$required"; then
        MISSING="$required"
        break
    fi
done
if [ -z "$MISSING" ]; then
    run_test "streaming text events" "PASS" ""
else
    run_test "streaming text events" "FAIL" "missing $MISSING"
fi
echo ""

# ── Test 8: streaming function tool emits function_call_arguments events ──
echo "--- Test 8: streaming function tool events ---"
EVENTS=$(curl -sf -N "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"mlx-serve",
    "input":"What is the weather in Tokyo? Use the get_weather tool.",
    "tools":[{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],
    "tool_choice":"required",
    "stream":true,
    "max_output_tokens":128,
    "temperature":0
  }' 2>/dev/null)

if echo "$EVENTS" | grep -qE "^event: response.function_call_arguments.delta" && \
   echo "$EVENTS" | grep -qE "^event: response.function_call_arguments.done" && \
   echo "$EVENTS" | grep -qE "^event: response.completed"; then
    run_test "streaming function_call events" "PASS" ""
else
    EVENT_ORDER=$(echo "$EVENTS" | grep -E "^event: " | awk '{print $2}' | tr '\n' ',')
    run_test "streaming function_call events" "FAIL" "events: $EVENT_ORDER"
fi
echo ""

# ── Test 11: previous_response_id not found ──
echo "--- Test 11: previous_response_id not found ---"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-serve","input":"hi","previous_response_id":"resp_does_not_exist","max_output_tokens":8}')
if [ "$HTTP_CODE" = "404" ]; then
    run_test "previous_response_id not found returns 404" "PASS" ""
else
    run_test "previous_response_id not found returns 404" "FAIL" "got HTTP $HTTP_CODE"
fi
echo ""

echo "=== Summary ==="
echo "Passed: $PASS / $TOTAL"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed: $FAIL"
    echo ""
    echo "Server log (last 40 lines):"
    tail -40 /tmp/mlx-serve-test-responses.log
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
