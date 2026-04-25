#!/bin/bash
# Integration test: verify response_format.json_schema is grammar-enforced.
#
# Without grammar enforcement, the model often emits prose around the JSON, or
# closes objects too early, especially under quantization. With enforcement, the
# token mask makes those outputs unreachable — every byte of the response must
# satisfy the schema.
#
# Usage: ./tests/test_json_schema.sh [model_dir] [port]

MODEL_DIR=${1:-~/.mlx-serve/models/gemma-4-e4b-it-4bit}
PORT=${2:-8099}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

echo "=== JSON Schema Constrained Generation Test ==="
echo "Model: $MODEL_DIR"
echo "Port: $PORT"
echo ""

echo "Starting server..."
./zig-out/bin/mlx-serve --model "$MODEL_DIR" --serve --port $PORT --log-level info 2>/tmp/mlx-serve-test-json-schema.log &
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

# ── Test 1: Simple object schema ──
echo "--- Test 1: Object with required string + integer ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "user", "content": "Make up a person record."}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer"}
          },
          "required": ["name", "age"],
          "additionalProperties": false
        }
      }
    }
  }')

CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  raw content: $CONTENT"

PARSED_OK=$(echo "$CONTENT" | python3 -c "
import sys, json
try:
    obj = json.loads(sys.stdin.read())
    assert isinstance(obj, dict)
    assert isinstance(obj.get('name'), str)
    assert isinstance(obj.get('age'), int)
    extra = set(obj.keys()) - {'name', 'age'}
    assert not extra, f'unexpected extra keys: {extra}'
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)

if [ "$PARSED_OK" = "ok" ]; then
    run_test "Object schema strictly satisfied" "PASS" ""
else
    run_test "Object schema strictly satisfied" "FAIL" "$PARSED_OK"
fi
echo ""

# ── Test 2: Enum constraint ──
echo "--- Test 2: Enum string ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "user", "content": "Pick a color."}
    ],
    "max_tokens": 32,
    "temperature": 0.7,
    "stream": false,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "color",
        "schema": {"type": "string", "enum": ["red", "green", "blue"]}
      }
    }
  }')

CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  raw content: $CONTENT"
ENUM_OK=$(echo "$CONTENT" | python3 -c "
import sys, json
try:
    v = json.loads(sys.stdin.read())
    assert v in ('red', 'green', 'blue'), f'not in enum: {v!r}'
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
if [ "$ENUM_OK" = "ok" ]; then
    run_test "Enum value picked from allowed set" "PASS" ""
else
    run_test "Enum value picked from allowed set" "FAIL" "$ENUM_OK"
fi
echo ""

# ── Test 3: Nested object with array ──
echo "--- Test 3: Nested object with array of strings ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "user", "content": "Make a recipe with name and 3 short ingredients."}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "recipe",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "ingredients": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["name", "ingredients"],
          "additionalProperties": false
        }
      }
    }
  }')

CONTENT=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  raw content: $CONTENT"
RECIPE_OK=$(echo "$CONTENT" | python3 -c "
import sys, json
try:
    obj = json.loads(sys.stdin.read())
    assert isinstance(obj.get('name'), str)
    items = obj.get('ingredients')
    assert isinstance(items, list) and all(isinstance(x, str) for x in items)
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
if [ "$RECIPE_OK" = "ok" ]; then
    run_test "Nested array-of-string schema satisfied" "PASS" ""
else
    run_test "Nested array-of-string schema satisfied" "FAIL" "$RECIPE_OK"
fi
echo ""

# ── Test 4: Streaming JSON schema ──
echo "--- Test 4: Streaming with json_schema produces valid JSON ---"
EVENTS=$(curl -sf -N "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [
      {"role": "user", "content": "Make up a book record."}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": true,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "book",
        "schema": {
          "type": "object",
          "properties": {
            "title": {"type": "string"},
            "year":  {"type": "integer"}
          },
          "required": ["title", "year"],
          "additionalProperties": false
        }
      }
    }
  }' 2>/dev/null)

ASSEMBLED=$(echo "$EVENTS" | python3 -c "
import sys, json
buf = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: '): continue
    payload = line[len('data: '):]
    if payload == '[DONE]': break
    try:
        chunk = json.loads(payload)
    except Exception:
        continue
    delta = chunk['choices'][0].get('delta', {})
    if 'content' in delta and delta['content'] is not None:
        buf.append(delta['content'])
print(''.join(buf))
")
echo "  assembled: $ASSEMBLED"
STREAM_OK=$(echo "$ASSEMBLED" | python3 -c "
import sys, json
try:
    obj = json.loads(sys.stdin.read())
    assert isinstance(obj.get('title'), str)
    assert isinstance(obj.get('year'), int)
    print('ok')
except Exception as e:
    print(f'fail:{e}')
" 2>/dev/null)
if [ "$STREAM_OK" = "ok" ]; then
    run_test "Streaming output assembles into valid JSON" "PASS" ""
else
    run_test "Streaming output assembles into valid JSON" "FAIL" "$STREAM_OK"
fi
echo ""

echo "=== Summary ==="
echo "Passed: $PASS / $TOTAL"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed: $FAIL"
    echo ""
    echo "Server log (last 50 lines):"
    tail -50 /tmp/mlx-serve-test-json-schema.log
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
