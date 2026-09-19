#!/bin/bash
# test_batched_features.sh — heterogeneous requests in ONE batched decode group.
# Every per-request feature is tested alone somewhere; none of them is tested
# beside the others in the same tick. Six requests with different contracts
# fire at once on a batching arch and each must keep ITS OWN: the tool call
# carries its own planted code, the schema request parses, the thinking
# request splits, logprobs describe content, the stop request cuts, and the
# short/long caps finish by their own max_tokens. The log must show the group
# actually formed (`[batched] ... engaged (slots=N)`, N >= 2) or the run
# proves nothing.
#
#   BATCH_FEAT_MODEL=<qwen3_5 pack> ./tests/test_batched_features.sh [port]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${BATCH_FEAT_MODEL:-$HOME/.mlx-serve/models/lmstudio-community/Qwen3.5-4B-MLX-4bit}"
PORT="${1:-11513}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
[[ -d "$MODEL" ]] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[[ -x "$BINARY" ]] || { echo "SKIP: $BINARY missing"; exit 0; }

WORK=$(mktemp -d "$HOME/claude-tmp/batched-feat.XXXXXX" 2>/dev/null || mktemp -d)
mkdir -p "$WORK/home"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 0.5
HOME="$WORK/home" "$BINARY" --serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    --log-level debug --max-concurrent 8 --prefix-cache-entries 0 > "$WORK/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf -m 2 "$BASE/health" >/dev/null || { echo "FAIL: server did not start"; tail -20 "$WORK/server.log"; exit 1; }
# warm the load so the group forms on the first tick of the burst
curl -s -m 300 "$BASE/v1/chat/completions" -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' >/dev/null

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $1 — ${2:-}"; }
J() { python3 -c "import sys,json; d=json.load(sys.stdin); print($1)" 2>/dev/null; }
post() { curl -s -m 600 -N "$BASE$1" -H 'Content-Type: application/json' --data-binary "$2"; }

CODE="ZQ7K$RANDOM"
TOOLS='[{"type":"function","function":{"name":"save_note","parameters":{"type":"object","properties":{"text":{"type":"string"},"tag":{"type":"string"}},"required":["text","tag"]}}}]'
SCHEMA='{"type":"json_schema","json_schema":{"name":"p","strict":true,"schema":{"type":"object","properties":{"animal":{"type":"string","enum":["cat","dog","owl"]},"legs":{"type":"integer"}},"required":["animal","legs"],"additionalProperties":false}}}'
LONG='Write a 250-word essay about the ocean.'

echo "=== burst: six contracts in one group ==="
post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Save a note with tag $CODE and text 'batched'. Use the save_note tool now.\"}],\"tools\":$TOOLS,\"stream\":true,\"max_tokens\":200,\"temperature\":0}" > "$WORK/tools.sse" & P1=$!
post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Describe an owl.\"}],\"response_format\":$SCHEMA,\"max_tokens\":80,\"temperature\":0}" > "$WORK/schema.json" & P2=$!
post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 12*12? Answer briefly.\"}],\"enable_thinking\":true,\"stream\":true,\"max_tokens\":400,\"temperature\":0}" > "$WORK/think.sse" & P3=$!
post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG\"}],\"logprobs\":true,\"top_logprobs\":3,\"max_tokens\":60,\"temperature\":0}" > "$WORK/logprobs.json" & P4=$!
post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to twenty in words, comma separated.\"}],\"stop\":[\"seven\"],\"stream\":true,\"max_tokens\":200,\"temperature\":0}" > "$WORK/stop.sse" & P5=$!
post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG\"}],\"max_tokens\":5,\"temperature\":0.7}" > "$WORK/cap5.json" & P6=$!
post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG\"}],\"max_tokens\":300,\"temperature\":0}" > "$WORK/cap300.json" & P7=$!
wait $P1 $P2 $P3 $P4 $P5 $P6 $P7

# engagement: the group really formed
SLOTS=$(grep -oE "\[batched\] .*engaged \(slots=[0-9]+" "$WORK/server.log" | grep -oE "[0-9]+$" | sort -n | tail -1)
[[ -n "$SLOTS" && "$SLOTS" -ge 2 ]] && ok "batched decode engaged (max slots=$SLOTS)" || bad "batched decode engaged" "no [batched] engaged line with slots>=2"

# tools (stream): the call carries its own code, args valid JSON
ARGS=$(grep '^data: {' "$WORK/tools.sse" | sed 's/^data: //' | python3 -c '
import sys,json
name=None; args=""
for line in sys.stdin:
    try: d=json.loads(line)
    except Exception: continue
    for tc in (d["choices"][0]["delta"].get("tool_calls") or []):
        f=tc.get("function",{}); name=f.get("name") or name; args+=f.get("arguments") or ""
print(name or ""); print(args)')
TNAME=$(echo "$ARGS" | head -1); TARGS=$(echo "$ARGS" | tail -n +2)
if [[ "$TNAME" == save_note ]]; then
    ok "tools: call names the declared tool"
    echo "$TARGS" | python3 -c 'import sys,json; json.load(sys.stdin)' 2>/dev/null && ok "tools: arguments valid JSON" || bad "tools: arguments valid JSON" "${TARGS:0:120}"
    [[ "$TARGS" == *"$CODE"* ]] && ok "tools: arguments carry this request's own code" || bad "tools: own code in arguments" "${TARGS:0:120}"
else
    bad "tools: call fired" "$(head -c 300 "$WORK/tools.sse")"
fi
grep -q '"finish_reason":"tool_calls"' "$WORK/tools.sse" && ok "tools: finish_reason tool_calls" || bad "tools: finish_reason tool_calls"
grep -q '^data: \[DONE\]' "$WORK/tools.sse" && ok "tools: [DONE]" || bad "tools: [DONE]"

# schema (non-stream): parses and conforms
C=$(J 'd["choices"][0]["message"]["content"]' < "$WORK/schema.json")
echo "$C" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d["animal"] in ("cat","dog","owl") and isinstance(d["legs"],int) and set(d)=={"animal","legs"}' 2>/dev/null \
    && ok "schema: content conforms in the group" || bad "schema: content conforms" "${C:0:160}"

# thinking (stream): reasoning and content both arrive, no tag leak in either
python3 - "$WORK/think.sse" <<'EOF' && ok "thinking: reasoning_content deltas + content, no tag leak" || bad "thinking: split in the group" "$(grep -c reasoning_content "$WORK/think.sse") reasoning chunks"
import sys,json
r=c=""
for line in open(sys.argv[1]):
    if not line.startswith("data: {"): continue
    d=json.loads(line[6:]); dl=d["choices"][0]["delta"]
    r+=dl.get("reasoning_content") or ""; c+=dl.get("content") or ""
assert r, "no reasoning"
assert "<think>" not in c and "</think>" not in c and "</think>" not in r, "tag leak"
EOF

# logprobs (non-stream): entries exist and rank 1 is the chosen token
python3 - "$WORK/logprobs.json" <<'EOF' && ok "logprobs: entries describe content, rank 1 == chosen" || bad "logprobs: in the group" "$(head -c 200 "$WORK/logprobs.json")"
import sys,json
d=json.load(open(sys.argv[1])); ch=d["choices"][0]; lp=ch["logprobs"]["content"]
assert len(lp)>0 and all(e["top_logprobs"][0]["token"]==e["token"] for e in lp)
assert "".join(e["token"] for e in lp).strip()==ch["message"]["content"].strip()
EOF

# stop (stream): content excludes the stop, finish stop
SC=$(grep '^data: {' "$WORK/stop.sse" | sed 's/^data: //' | python3 -c '
import sys,json; out=""
for l in sys.stdin:
    try: out+=json.loads(l)["choices"][0]["delta"].get("content") or ""
    except Exception: pass
print(out)')
[[ "$SC" == *six* && "$SC" != *seven* ]] && ok "stop: cut before the stop sequence in the group" || bad "stop: cut" "${SC:0:120}"
grep -q '"finish_reason":"stop"' "$WORK/stop.sse" && ok "stop: finish_reason stop" || bad "stop: finish_reason stop"

# caps: each finishes by its OWN max_tokens
[[ "$(J 'd["usage"]["completion_tokens"]' < "$WORK/cap5.json")" -le 5 && "$(J 'd["choices"][0]["finish_reason"]' < "$WORK/cap5.json")" =~ ^(length|stop)$ ]] && ok "cap 5: <=5 tokens" || bad "cap 5" "$(head -c 200 "$WORK/cap5.json")"
N300=$(J 'd["usage"]["completion_tokens"]' < "$WORK/cap300.json")
[[ -n "$N300" && "$N300" -gt 100 && "$N300" -le 300 ]] && ok "cap 300: long answer ($N300 tokens)" || bad "cap 300" "$(head -c 200 "$WORK/cap300.json")"

echo "=== after: alive, no error ==="
curl -sf -m 5 "$BASE/health" >/dev/null && ok "server alive" || bad "server alive"
grep -qE "\[mlx\] error|panic|Segmentation" "$WORK/server.log" && bad "no MLX error / crash line" || ok "no MLX error / crash line"

echo
echo "batched-features: $PASS passed, $FAIL failed  (artifacts: $WORK)"
[[ "$FAIL" -eq 0 ]] && rm -rf "$WORK"
[[ "$FAIL" -eq 0 ]]
