#!/bin/bash
# test_api_edges.sh — request-validation edges across the four text surfaces
# and Ollama, on one small model. Every check is a server CONTRACT (status +
# shape), never a checkpoint expectation: what a request that is malformed,
# out of range, or unsupported gets back, and that the server is still alive.
#
#   ./tests/test_api_edges.sh [model_dir] [port]
#
# Found on first run (2026-09-16): top_p 0 masked every token (uniform garbage),
# stop "" matched at position 0 (empty reply), json_schema without a schema
# fell open silently, an undecodable image_url vanished from the prompt, an
# empty embedding input was a 500, and Ollama's load handshake was a 400.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
PORT="${2:-11512}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
[[ -d "$MODEL" ]] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[[ -x "$BINARY" ]] || { echo "SKIP: $BINARY missing"; exit 0; }

WORK=$(mktemp -d "$HOME/claude-tmp/api-edges.XXXXXX" 2>/dev/null || mktemp -d)
mkdir -p "$WORK/home"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 0.5
HOME="$WORK/home" "$BINARY" --serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    --log-level debug --max-concurrent 4 > "$WORK/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT
for i in $(seq 1 90); do curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf -m 2 "$BASE/health" >/dev/null || { echo "FAIL: server did not start"; tail -20 "$WORK/server.log"; exit 1; }

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $1 — ${2:-}"; }
# status <expected> <name> <method> <path> [body]  → also stashes the body in $R
R=""
req() {
    local method="$1" path="$2" body="${3:-}"
    if [[ -n "$body" ]]; then
        R=$(curl -s -m 120 -o "$WORK/body" -w '%{http_code}' -X "$method" "$BASE$path" -H 'Content-Type: application/json' --data-binary "$body")
    else
        R=$(curl -s -m 120 -o "$WORK/body" -w '%{http_code}' -X "$method" "$BASE$path")
    fi
    BODY=$(cat "$WORK/body")
}
J() { python3 -c "import sys,json; d=json.load(sys.stdin); print($1)" 2>/dev/null; }
expect_status() { # <want> <name>
    if [[ "$R" == "$1" ]]; then ok "$2: $1"; else bad "$2: want $1 got $R" "$(echo "$BODY" | head -c 160)"; fi
}
U='{"role":"user","content":"hi"}'

echo "=== request validation ==="
req POST /v1/chat/completions '';                                   expect_status 400 "empty body"
req POST /v1/chat/completions '{not json';                          expect_status 400 "invalid JSON"
req POST /v1/chat/completions '[1,2]';                              expect_status 400 "array body"
req POST /v1/chat/completions '{"model":"m"}';                      expect_status 400 "no messages"
req POST /v1/chat/completions '{"model":"m","messages":[]}';        expect_status 400 "empty messages"
req POST /v1/chat/completions '{"model":"m","messages":"x"}';       expect_status 400 "messages not array"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"n\":2}"; expect_status 400 "n>1"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"x":'"$(python3 -c 'print("["*3000+"]"*3000)')"'}'
expect_status 200 "3000-deep nesting in an ignored field"
req GET  /v1/chat/completions;                                      expect_status 404 "GET on a POST route"
req POST /v1/bogus '{"model":"m"}';                                 expect_status 404 "unknown route"
req OPTIONS /v1/chat/completions;                                   expect_status 204 "CORS preflight"

echo "=== sampling edges ==="
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"top_p\":0,\"temperature\":1.0,\"max_tokens\":8}"
GREEDY=$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"temperature\":0,\"max_tokens\":8}"
T0=$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')
[[ -n "$GREEDY" && "$GREEDY" == "$T0" ]] && ok "top_p 0 is greedy (== temperature 0)" || bad "top_p 0 is greedy" "top_p0='$GREEDY' temp0='$T0'"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"top_k\":1,\"temperature\":1.0,\"max_tokens\":8}"
[[ "$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')" == "$T0" ]] && ok "top_k 1 is greedy" || bad "top_k 1 is greedy"
for f in '"temperature":-1' '"temperature":100' '"top_p":2' '"top_k":-3' '"seed":9223372036854775808' '"seed":"abc"' '"max_tokens":0' '"max_tokens":-5' '"logit_bias":{"1":-100}' '"presence_penalty":2,"frequency_penalty":2'; do
    [[ "$f" == '"max_tokens"'* ]] && cap="" || cap=',"max_tokens":5'
    req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],$f$cap}"
    [[ "$R" == 200 && -n "$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')" ]] && ok "out-of-range $f answers" || bad "out-of-range $f" "$R $(echo "$BODY" | head -c 120)"
done

echo "=== stop sequences ==="
CNT='{"role":"user","content":"Count from one to ten in words, comma separated."}'
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$CNT],\"stop\":\"four\",\"max_tokens\":60,\"temperature\":0}"
C=$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]'); FR=$(echo "$BODY" | J 'd["choices"][0]["finish_reason"]')
[[ "$C" != *four* && "$FR" == stop ]] && ok "stop string cuts before the match, finish stop" || bad "stop string" "fr=$FR c=${C:0:80}"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"stop\":\"\",\"max_tokens\":5}"
[[ -n "$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')" ]] && ok "empty stop string is ignored" || bad "empty stop string is ignored" "$(echo "$BODY" | head -c 160)"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"stop\":[\"\",\"zzz\"],\"max_tokens\":5}"
[[ -n "$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')" ]] && ok "empty stop entry in an array is ignored" || bad "empty stop entry in an array" "$(echo "$BODY" | head -c 160)"
req POST /v1/messages "{\"model\":\"m\",\"max_tokens\":60,\"stop_sequences\":[\"four\"],\"messages\":[$CNT],\"temperature\":0}"
[[ "$(echo "$BODY" | J 'd["stop_reason"]')" == stop_sequence && "$(echo "$BODY" | J 'd["stop_sequence"]')" == four ]] && ok "/v1/messages echoes the matched stop_sequence" || bad "/v1/messages stop_sequence echo" "$(echo "$BODY" | head -c 160)"

echo "=== structured output ==="
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"x\",\"schema\":\"notaschema\"}},\"max_tokens\":20}"
expect_status 400 "json_schema with a non-object schema"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"response_format\":{\"type\":\"json_schema\"},\"max_tokens\":20}"
expect_status 400 "json_schema without a schema"
req POST /v1/messages "{\"model\":\"m\",\"max_tokens\":20,\"output_config\":{\"format\":{\"type\":\"json_schema\"}},\"messages\":[$U]}"
expect_status 400 "messages: output_config json_schema without a schema"
req POST /v1/responses '{"model":"m","max_output_tokens":20,"text":{"format":{"type":"json_schema","name":"x","schema":"nope"}},"input":"hi"}'
expect_status 400 "responses: text.format json_schema with a non-object schema"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"Describe a cat."}],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object","properties":{"kind":{"type":"string","enum":["cat","dog"]},"legs":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}},"meta":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}},"required":["kind","legs","tags","meta"],"additionalProperties":false}}},"max_tokens":120,"temperature":0}'
C=$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')
echo "$C" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d["kind"] in ("cat","dog") and isinstance(d["legs"],int) and isinstance(d["tags"],list) and isinstance(d["meta"]["ok"],bool) and set(d)=={"kind","legs","tags","meta"}' 2>/dev/null \
    && ok "enum + nested object + array schema is enforced" || bad "nested schema enforced" "${C:0:160}"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"Give a JSON object with keys a and b."}],"response_format":{"type":"json_object"},"max_tokens":60,"temperature":0}'
echo "$BODY" | J 'd["choices"][0]["message"]["content"]' | python3 -c 'import sys,json; json.load(sys.stdin)' 2>/dev/null && ok "json_object yields parseable JSON" || bad "json_object parseable"

echo "=== media on the wire ==="
IMG_BAD='{"type":"image_url","image_url":{"url":"data:image/png;base64,!!!notbase64"}}'
IMG_HTTP='{"type":"image_url","image_url":{"url":"http://127.0.0.1:1/x.png"}}'
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"what\"},$IMG_BAD]}],\"max_tokens\":5}"
expect_status 400 "chat: undecodable image_url refused"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":[$IMG_HTTP]}],\"max_tokens\":5}"
expect_status 400 "chat: remote image URL refused by name"
[[ "$(echo "$BODY" | J 'd["error"]["message"]')" == *"remote URLs are not fetched"* ]] && ok "chat: refusal names the cause" || bad "chat: refusal names the cause" "$(echo "$BODY" | head -c 160)"
req POST /v1/messages '{"model":"m","max_tokens":5,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}'
expect_status 400 "messages: undecodable image block refused"
req POST /v1/responses '{"model":"m","input":[{"role":"user","content":[{"type":"input_image","image_url":"http://127.0.0.1:1/x.png"}]}],"max_output_tokens":5}'
expect_status 400 "responses: remote input_image refused"
# a historical (non-active) image is never decoded, so it is never refused
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":[$IMG_HTTP,{\"type\":\"text\",\"text\":\"old\"}]},{\"role\":\"assistant\",\"content\":\"ok\"},$U],\"max_tokens\":5}"
expect_status 200 "chat: a historical bad image is not the active turn's problem"

echo "=== content shapes ==="
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"Repeat exactly: ALPHA"},{"type":"text","text":" BRAVO"}]}],"max_tokens":10,"temperature":0}'
[[ "$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')" == *ALPHA*BRAVO* ]] && ok "text parts join in order" || bad "text parts join in order" "$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"Line1 Line2. Reply with the word DONE."}],"max_tokens":8,"temperature":0}'
[[ "$R" == 200 && "$(echo "$BODY" | J 'd["usage"]["prompt_tokens"]')" -gt 15 ]] && ok "NUL byte does not truncate the prompt" || bad "NUL byte prompt" "$R $(echo "$BODY" | head -c 120)"
req POST /v1/chat/completions "$(printf '{"model":"m","messages":[{"role":"user","content":"Repeat exactly: \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e \xf0\x9f\x8d\xa3"}],"max_tokens":12,"temperature":0}')"
C=$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')
[[ "$C" == *日本語* || "$C" == *🍣* ]] && ok "CJK + emoji round-trip through the escaper" || bad "CJK + emoji" "$C"
req POST /v1/chat/completions "$(printf '{"model":"m","messages":[{"role":"user","content":"bad \xff\xfe bytes"}],"max_tokens":5}')"
[[ "$R" == 400 || "$R" == 200 ]] && ok "invalid UTF-8 body is answered ($R), not dropped" || bad "invalid UTF-8 body" "$R"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"developer","content":"Always answer in uppercase."},{"role":"user","content":"say hello"}],"max_tokens":10,"temperature":0}'
expect_status 200 "developer role accepted"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"Write the alphabet."},{"role":"assistant","content":"A, B, C, D,"}],"continue_final_message":true,"max_tokens":8,"temperature":0}'
C=$(echo "$BODY" | J 'd["choices"][0]["message"]["content"]')
[[ "$C" != *"A, B"* ]] && ok "continue_final_message returns only the continuation" || bad "continuation echoes the prefix" "$C"

echo "=== embeddings + tokenize ==="
req POST /v1/embeddings '{"model":"m","input":""}';                   expect_status 400 "empty embedding input"
req POST /v1/embeddings '{"model":"m","input":["a",""]}';             expect_status 400 "empty entry in embedding array"
req POST /v1/embeddings '{"model":"m","input":["hello","world"]}'
[[ "$R" == 200 && "$(echo "$BODY" | J 'len(d["data"])')" == 2 ]] && ok "embedding array answers one vector per input" || bad "embedding array" "$R"
req POST /tokenize '{"content":"hello world"}'
[[ "$(echo "$BODY" | J 'len(d["tokens"])')" -ge 2 ]] && ok "/tokenize" || bad "/tokenize" "$BODY"

echo "=== Ollama ==="
req POST /api/generate '{"model":"m"}'
[[ "$R" == 200 && "$(echo "$BODY" | J 'd["done"]')" == True && "$(echo "$BODY" | J 'd["done_reason"]')" == load ]] && ok "/api/generate with no prompt is the load handshake" || bad "/api/generate load handshake" "$R $(echo "$BODY" | head -c 160)"
req POST /api/generate '{"model":"m","prompt":"The capital of France is","raw":true,"stream":false,"options":{"num_predict":5,"temperature":0}}'
[[ "$R" == 200 && -n "$(echo "$BODY" | J 'd["response"]')" ]] && ok "/api/generate raw" || bad "/api/generate raw" "$R"
req POST /api/chat '{"model":"m","messages":[{"role":"user","content":"weather in Paris?"}],"stream":false,"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"options":{"num_predict":60,"temperature":0}}'
[[ "$R" == 200 ]] && ok "/api/chat with tools answers" || bad "/api/chat tools" "$R"
req POST /api/show '{"name":"zzz"}';                                   expect_status 404 "/api/show unknown model"

echo "=== Responses API ==="
req POST /v1/responses '{"model":"m"}';                                expect_status 400 "responses: no input"
req POST /v1/responses '{"model":"m","input":"hi","background":true}';  expect_status 400 "responses: background"
req POST /v1/responses '{"model":"m","input":"hi","previous_response_id":"resp_nope"}'; expect_status 404 "responses: unknown previous_response_id"
req POST /v1/responses '{"model":"m","input":"Give a person with name and age.","text":{"format":{"type":"json_schema","name":"p","schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}}},"max_output_tokens":60,"temperature":0}'
echo "$BODY" | python3 -c 'import sys,json; d=json.load(sys.stdin); t=[c["text"] for o in d["output"] if o["type"]=="message" for c in o["content"]][0]; j=json.loads(t); assert isinstance(j["age"],int)' 2>/dev/null && ok "responses: text.format json_schema enforced" || bad "responses json_schema" "$(echo "$BODY" | head -c 160)"
req POST /v1/messages "{\"model\":\"m\",\"messages\":[$U]}";           expect_status 400 "messages: max_tokens required"

echo "=== alive ==="
req GET /health; expect_status 200 "server alive after every edge"
grep -qE "\[mlx\] error|panic|Segmentation" "$WORK/server.log" && bad "no MLX error / crash line in the log" || ok "no MLX error / crash line in the log"

echo
echo "api-edges: $PASS passed, $FAIL failed  (log: $WORK/server.log)"
[[ "$FAIL" -eq 0 ]] && rm -rf "$WORK"
[[ "$FAIL" -eq 0 ]]
