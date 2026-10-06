#!/bin/bash
# `tool_choice` that obliges a call is ENFORCED, not asked for: the prompt tells
# the model not to call anything, and the reply must still be a call (the
# tool-eval-bench probe). After the thought closes the server commits the
# family's tool-call opener; the model writes the call from there.
#
# Checks: chat required (non-stream + stream, and at a max_tokens the thought
# outlasts), chat named function, Anthropic `any`; each returns a call with
# non-empty arguments. A named function the request does not declare is a 400
# on chat, messages and responses.
#
# Usage: ./tests/test_tool_choice_required.sh [model_dir] [port]
set -u

source "$(dirname "$0")/_lib_models.sh"
MODEL="${1:-$(find_model mlx-community/Qwen3.5-4B-MLX-4bit mlx-community/Qwen3.5-0.8B-MLX-4bit)}"
PORT="${2:-11268}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0
FAIL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

check() {
    if [ "$2" = "1" ]; then PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1";
    else FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; fi
}

if [ ! -d "$MODEL" ]; then echo "skip: model not found ($MODEL)"; exit 0; fi

LOG=$(mktemp); OUT=$(mktemp)
"$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f "$LOG" "$OUT"' EXIT
for _ in $(seq 1 120); do sleep 2; curl -sf "$BASE/health" >/dev/null 2>&1 && break; done
if ! curl -sf "$BASE/health" >/dev/null 2>&1; then echo "server failed to start"; tail -5 "$LOG"; exit 1; fi

TOOLS='[{"type":"function","function":{"name":"probe_ping","description":"Acknowledge the probe. Call this with any value.","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}}},
        {"type":"function","function":{"name":"calculator","description":"Evaluate math","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}]'
MSG='[{"role":"user","content":"Reply with the single word OK. Do not call any tools."}]'

chat() { curl -s -m 300 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"x\",\"messages\":$MSG,\"tools\":$TOOLS,\"temperature\":0,\"max_tokens\":${2:-1024},$1}"; }

echo "[tool-choice] === $(basename "$MODEL") ==="

chat '"tool_choice":"required"' > "$OUT"
check "chat required: a call with arguments" "$(python3 -c '
import json,sys
m=json.load(open(sys.argv[1]))["choices"][0]["message"]
c=m.get("tool_calls") or []
print(1 if c and json.loads(c[0]["function"]["arguments"]) else 0)' "$OUT" 2>/dev/null)"

chat '"tool_choice":"required","stream":true' > "$OUT"
check "chat required, stream: a call with arguments" "$(python3 -c '
import json,sys
args=""
for l in open(sys.argv[1]):
    if not l.startswith("data: ") or "[DONE]" in l: continue
    for tc in json.loads(l[6:])["choices"][0]["delta"].get("tool_calls") or []:
        args+=tc.get("function",{}).get("arguments","")
print(1 if args and json.loads(args) else 0)' "$OUT" 2>/dev/null)"

# A thought longer than three quarters of max_tokens: the server closes it for the call.
LONG='[{"role":"user","content":"Before you answer, think carefully through the first twenty prime numbers one by one. Then reply with the single word OK. Do not call any tools."}]'
curl -s -m 300 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"x\",\"messages\":$LONG,\"tools\":$TOOLS,\"temperature\":0,\"max_tokens\":160,\"enable_thinking\":true,\"tool_choice\":\"required\"}" > "$OUT"
check "chat required, thought cut by the deadline: a call with arguments" "$(python3 -c '
import json,sys
c=json.load(open(sys.argv[1]))["choices"][0]["message"].get("tool_calls") or []
print(1 if c and json.loads(c[0]["function"]["arguments"]) else 0)' "$OUT" 2>/dev/null)"
check "deadline engagement logged" "$(grep -q 'thought closed for it' "$LOG" && echo 1 || echo 0)"

chat '"tool_choice":{"type":"function","function":{"name":"calculator"}}' > "$OUT"
check "chat named: calls calculator" "$(python3 -c '
import json,sys
c=json.load(open(sys.argv[1]))["choices"][0]["message"].get("tool_calls") or []
print(1 if c and c[0]["function"]["name"]=="calculator" else 0)' "$OUT" 2>/dev/null)"

curl -s -m 300 "$BASE/v1/messages" -H 'content-type: application/json' -d "{\"model\":\"x\",\"max_tokens\":1024,\"temperature\":0,
  \"messages\":$MSG,\"tool_choice\":{\"type\":\"any\"},
  \"tools\":[{\"name\":\"probe_ping\",\"description\":\"Acknowledge the probe.\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"value\":{\"type\":\"string\"}},\"required\":[\"value\"]}}]}" > "$OUT"
check "messages any: a tool_use block" "$(python3 -c '
import json,sys
b=[x for x in json.load(open(sys.argv[1]))["content"] if x["type"]=="tool_use"]
print(1 if b and b[0]["input"] else 0)' "$OUT" 2>/dev/null)"

code() { curl -s -o /dev/null -w '%{http_code}' -m 60 "$BASE$1" -H 'content-type: application/json' -d "$2"; }
# Bodies go in variables first: a JSON body inline in a nested $( ) splits.
B="{\"model\":\"x\",\"messages\":$MSG,\"tools\":$TOOLS,\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"bogus\"}}}"
R=$(code /v1/chat/completions "$B"); check "chat: an undeclared named function is a 400" "$([ "$R" = 400 ] && echo 1 || echo 0)"
B="{\"model\":\"x\",\"max_tokens\":64,\"messages\":$MSG,\"tools\":[{\"name\":\"probe_ping\",\"input_schema\":{\"type\":\"object\"}}],\"tool_choice\":{\"type\":\"tool\",\"name\":\"bogus\"}}"
R=$(code /v1/messages "$B"); check "messages: an undeclared named tool is a 400" "$([ "$R" = 400 ] && echo 1 || echo 0)"
B="{\"model\":\"x\",\"input\":\"hi\",\"tools\":[{\"type\":\"function\",\"name\":\"probe_ping\",\"parameters\":{\"type\":\"object\"}}],\"tool_choice\":{\"type\":\"function\",\"name\":\"bogus\"}}"
R=$(code /v1/responses "$B"); check "responses: an undeclared named function is a 400" "$([ "$R" = 400 ] && echo 1 || echo 0)"

check "engagement logged" "$(grep -q '\[tool-choice\] call opener forced' "$LOG" && echo 1 || echo 0)"

echo "[tool-choice] $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
