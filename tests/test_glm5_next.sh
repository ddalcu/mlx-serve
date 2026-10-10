#!/bin/bash
# GLM-5.3-Flash (glm5_next) live end-to-end on an MLX pack (TensorFold's GLM-5.3-Flash-MLX-*):
#
#   GLM5_PACK=~/.mlx-serve/models/TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP ./tests/test_glm5_next.sh
#   (add GLM5_DRAFTER=~/.mlx-serve/models/incoai/GLM-5.3-Flash-DFlash2 for [10])
#
#   [0] advertised as glm5_next      [4] tool round-trip renders the template, not the fallback
#   [1] short answer, thinking off   [5] needle past the indexer's 2048-token budget (sparse path)
#   [2] thinking on by default, and  [6] prefix reuse: cached tokens, same answer
#       low effort thinks less       [7] streaming carries no think/tool markup
#   [3] parallel tool calls             [8] the pack's MTP head loads and drafts
#                                       [9] concurrent requests share one batched forward
#   [10] with GLM5_DRAFTER=<incoai/GLM-5.3-Flash-DFlash2> (second boot): the DFlash2 round policy engages on
#        predictable text, stays armed on novel text (plain rounds, never the sticky serial fallback), and
#        leaves the answer, the stream, a tool call and concurrent requests unchanged
#
# Hermetic counterparts: the config-parse tests in model.zig, the effort and tool-history
# render tests in chat.zig, and the `glm5_next fixture` + `glm5_next MTP fixture` parity
# tests in transformer.zig.

set -euo pipefail

MODEL="${GLM5_PACK:-}"
if [ -z "$MODEL" ]; then echo "SKIP: GLM5_PACK not set"; exit 0; fi
if [ ! -f "$MODEL/config.json" ]; then echo "FAIL: $MODEL/config.json not found"; exit 1; fi

PORT="${GLM5_TEST_PORT:-11361}"
BASE="http://127.0.0.1:$PORT"
BIN="$(dirname "$0")/../zig-out/bin/mlx-serve"
LOG=$(mktemp /tmp/glm5_test_serve.XXXXXX)
SCRATCH_HOME=$(mktemp -d /tmp/glm5_test_home.XXXXXX)

HOME="$SCRATCH_HOME" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 32768 > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$SCRATCH_HOME"; }
trap cleanup EXIT

for _ in $(seq 1 300); do
    grep -q "Model ready (loaded on inference thread)" "$LOG" && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1; fi
    sleep 3
done
grep -q "Model ready (loaded on inference thread)" "$LOG" || { echo "FAIL: model did not load"; tail -20 "$LOG"; exit 1; }

pass=0; fail=0
check() { if grep -qF "$3" <<< "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1"; echo "  wanted: $3"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); fi; }
check_absent() { if grep -qF "$3" <<< "$2"; then echo "FAIL $1 ('$3' present)"; echo "  got: $(echo "$2" | head -c 400)"; fail=$((fail+1)); else echo "PASS $1"; pass=$((pass+1)); fi; }
chat() { curl -s -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$1"; }
reasoning_len() { python3 -c 'import json,sys; print(len(json.load(sys.stdin)["choices"][0]["message"].get("reasoning_content") or ""))'; }
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

M=$(curl -s -m 30 "$BASE/v1/models")
check "[0] advertised as glm5_next" "$M" '"architecture":"glm5_next"'

R1=$(chat '{"max_tokens":60,"temperature":0,"enable_thinking":false,"messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}]}')
check "[1] answers Paris" "$R1" "Paris"
check_absent "[1] no reasoning" "$R1" '"reasoning_content"'

Q='A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. A second ball costs half as much as the first ball. How much do the bat and both balls cost together, in cents?'
R2=$(chat '{"max_tokens":4000,"temperature":0,"messages":[{"role":"user","content":"'"$Q"'"}]}')
check "[2] reasoning_content by default" "$R2" '"reasoning_content"'
check "[2] answer carries 112.5" "$R2" "112.5"
check_absent "[2] no think tags" "$R2" "</think>"
LOW=$(chat '{"max_tokens":4000,"temperature":0,"reasoning_effort":"low","messages":[{"role":"user","content":"'"$Q"'"}]}' | reasoning_len)
MAX=$(echo "$R2" | reasoning_len)
[ "$LOW" -lt "$MAX" ] && { echo "PASS [2] low effort thinks less ($LOW < $MAX chars)"; pass=$((pass+1)); } || { echo "FAIL [2] low effort $LOW >= default $MAX"; fail=$((fail+1)); }

T=$(chat '{"max_tokens":2000,"temperature":0,"messages":[{"role":"user","content":"What is the weather in Paris and in Tokyo right now? Use the tool."}],"tools":'"$TOOLS"'}')
check "[3] tool calls emitted" "$T" '"tool_calls"'
check "[3] Paris call" "$T" 'Paris'
check "[3] Tokyo call" "$T" 'Tokyo'
check_absent "[3] no tool markup" "$T" "<tool_call>"
check_absent "[3] no arg markup" "$T" "<arg_key>"

RT=$(chat '{"max_tokens":2000,"temperature":0,"messages":[
  {"role":"user","content":"What is the weather in Paris? Use the tool."},
  {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},
  {"role":"tool","tool_call_id":"c1","content":"{\"temp_c\": 21, \"conditions\": \"partly cloudy\"}"}],"tools":'"$TOOLS"'}')
check "[4] round-trip answer uses the result" "$RT" "21"
check_absent "[4] no fallback turn markup" "$RT" "<end_of_turn>"
check_absent "[4] template rendered" "$(cat "$LOG")" "jinja render failed"

NEEDLE=$(python3 - <<'PY'
import json
filler = " ".join(f"Section {i}: the weather report says clouds drift over hills and rivers flow to the sea." for i in range(400))
msg = "The vault combination for the Aldergate safe is 74-19-52.\n\n" + filler + "\n\nWhat is the vault combination for the Aldergate safe? Answer with just the digits."
print(json.dumps({"max_tokens": 200, "temperature": 0, "enable_thinking": False, "messages": [{"role": "user", "content": msg}]}))
PY
)
DIGITS=$(chat "$NEEDLE" | python3 -c 'import json,re,sys; print(re.sub(r"\D", "", json.load(sys.stdin)["choices"][0]["message"]["content"]))')
check "[5] needle recovered at ~8k tokens" "$DIGITS" "741952"

REQ6='{"max_tokens":60,"temperature":0,"enable_thinking":false,"messages":[{"role":"user","content":"My dog is called Biscuit and I like teal. What is my dog called?"}]}'
A=$(chat "$REQ6"); B=$(chat "$REQ6")
check "[6] first answer" "$A" "Biscuit"
CA=$(echo "$A" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
CB=$(echo "$B" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
[ "$CA" = "$CB" ] && { echo "PASS [6] same answer after reuse"; pass=$((pass+1)); } || { echo "FAIL [6] answer changed after reuse"; fail=$((fail+1)); }
CACHED=$(echo "$B" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["prompt_tokens_details"]["cached_tokens"])')
[ "$CACHED" -gt 0 ] && { echo "PASS [6] prefix cache engaged ($CACHED tokens)"; pass=$((pass+1)); } || { echo "FAIL [6] 0 cached tokens"; fail=$((fail+1)); }

S=$(curl -s -N -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{"stream":true,"max_tokens":2000,"temperature":0,
  "messages":[{"role":"user","content":"What is 15% of 80? Brief."}]}' | grep '^data: {' | python3 -c '
import json, sys
r, c = [], []
for line in sys.stdin:
    for ch in json.loads(line[6:]).get("choices", []):
        d = ch.get("delta", {})
        r.append(d.get("reasoning_content") or ""); c.append(d.get("content") or "")
print("REASONING:" + "".join(r)); print("CONTENT:" + "".join(c))')
check "[7] streamed answer carries 12" "$S" "12"
check_absent "[7] no think tags streamed" "$S" "<think"
check_absent "[7] no tool markup streamed" "$S" "<tool_call"

check "[8] MTP head loaded" "$(cat "$LOG")" "GLM MTP head ready"
check "[8] MTP drafted" "$(cat "$LOG")" "[spec-stats] mode=mtp"

# [9] Concurrent requests decode in ONE batched forward (rows of a window; each slot's KDA state and
# DSA cache advance on their own). Three counting streams of different lengths (so the group shrinks
# mid-flight) beside one carrying ~8k tokens (the sparse selection runs under batching).
CONC=$(python3 - "$PORT" <<'PY'
import json, sys, threading, urllib.request
port = sys.argv[1]
filler = " ".join(f"Section {i}: the weather report says clouds drift over hills and rivers flow to the sea." for i in range(400))
needle = "The vault combination for the Aldergate safe is 74-19-52.\n\n" + filler + "\n\nWhat is the vault combination for the Aldergate safe? Answer with just the digits."
reqs = [
    ("count30", "Count from 1 to 30, separated by single spaces, and write nothing else.", 150),
    ("count45", "Count from 1 to 45, separated by single spaces, and write nothing else.", 220),
    ("count60", "Count from 1 to 60, separated by single spaces, and write nothing else.", 300),
    ("741952", needle, 200),
]
out = {}
def run(i):
    key, msg, mt = reqs[i]
    body = {"max_tokens": mt, "temperature": 0, "enable_thinking": False, "messages": [{"role": "user", "content": msg}]}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    c = json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["message"]["content"]
    out[i] = "".join(ch for ch in c if ch.isdigit()) if key.isdigit() else c.split()
ts = [threading.Thread(target=run, args=(i,)) for i in range(len(reqs))]
[t.start() for t in ts]; [t.join() for t in ts]
def counted(toks, n):  # the stream counted 1..n in order
    return toks[:n] == [str(k) for k in range(1, n + 1)]
print("count30=%s count45=%s count60=%s needle=%s" % (counted(out[0], 30), counted(out[1], 45), counted(out[2], 60), out[3]))
PY
)
check "[9] 30-count stream correct under batching" "$CONC" "count30=True"
check "[9] 45-count stream correct under batching" "$CONC" "count45=True"
check "[9] 60-count stream correct under batching" "$CONC" "count60=True"
check "[9] needle recovered under batching" "$CONC" "needle=741952"
check "[9] batched forward engaged" "$(cat "$SCRATCH_HOME"/.mlx-serve/logs/*.log 2>/dev/null)" "[batched] glm batched decode engaged"

check_absent "[log] no MLX error" "$(cat "$LOG")" "[mlx]"

# [10] The DFlash2 drafter on top. Its own boot: the first server's memory goes back first, the pack is 170 GB.
if [ -n "${GLM5_DRAFTER:-}" ]; then
    [ -f "$GLM5_DRAFTER/config.json" ] || { echo "FAIL: $GLM5_DRAFTER/config.json not found"; exit 1; }
    kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true
    LOG1="$LOG"; PORT=$((PORT + 1)); BASE="http://127.0.0.1:$PORT"; LOG=$(mktemp /tmp/glm5_test_serve_dflash.XXXXXX)
    HOME="$SCRATCH_HOME" "$BIN" --model "$MODEL" --drafter "$GLM5_DRAFTER" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 32768 > "$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 300); do
        grep -q "Model ready (loaded on inference thread)" "$LOG" && break
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "FAIL: DFlash server died during load"; tail -20 "$LOG"; exit 1; fi
        sleep 3
    done
    grep -q "Model ready (loaded on inference thread)" "$LOG" || { echo "FAIL: DFlash server did not load"; tail -20 "$LOG"; exit 1; }
    check "[10] DFlash2 drafter loaded" "$(cat "$LOG")" "DFlash drafter ready"

    content() { python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'; }
    # The request's own [spec-stats] line is the newest one once it has finished.
    last_stats() { sleep 1; grep -F "[spec-stats] mode=dflash" "$LOG" | tail -1; }
    CODE='{"max_tokens":300,"temperature":0,"enable_thinking":false,"messages":[{"role":"user","content":"Write a Python function that checks whether a string is a palindrome, with a docstring and three unit tests. Output only the code."}]}'
    CODE_OFF='{"max_tokens":300,"temperature":0,"enable_thinking":false,"enable_drafter":false,"enable_mtp":false,"messages":[{"role":"user","content":"Write a Python function that checks whether a string is a palindrome, with a docstring and three unit tests. Output only the code."}]}'

    D1=$(chat "$CODE" | content); ST1=$(last_stats)
    check "[10] predictable text drafts and verifies" "$ST1" "mode=dflash attempts="
    case "$ST1" in *"attempts=0 "*) echo "FAIL [10] no round was ever drafted on predictable text"; fail=$((fail+1));; *) echo "PASS [10] rounds were drafted on predictable text"; pass=$((pass+1));; esac
    check "[10] per-position acceptance is reported" "$ST1" "acc_pos="
    check "[10] verify widths are reported" "$ST1" "rows_hist="
    check "[10] the policy never falls back for good" "$ST1" "runtime_disabled=false"

    D_OFF=$(chat "$CODE_OFF" | content)
    [ "${D1:0:150}" = "${D_OFF:0:150}" ] && { echo "PASS [10] drafter on == plain serial over the first 150 characters (greedy)"; pass=$((pass+1)); } || { echo "FAIL [10] greedy answer moved with the drafter"; echo "  on : ${D1:0:150}"; echo "  off: ${D_OFF:0:150}"; fail=$((fail+1)); }

    SD=$(curl -s -N -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$(echo "$CODE" | python3 -c 'import json,sys; d=json.load(sys.stdin); d["stream"]=True; print(json.dumps(d))')" | grep '^data: {' | python3 -c '
import json, sys
c = []
for line in sys.stdin:
    for ch in json.loads(line[6:]).get("choices", []):
        c.append(ch.get("delta", {}).get("content") or "")
print("".join(c))')
    [ "${SD:0:150}" = "${D1:0:150}" ] && { echo "PASS [10] stream == non-stream over the first 150 characters"; pass=$((pass+1)); } || { echo "FAIL [10] streamed answer differs from the non-streamed one"; echo "  stream: ${SD:0:150}"; echo "  plain : ${D1:0:150}"; fail=$((fail+1)); }

    NOVEL=$(chat '{"max_tokens":400,"temperature":0,"enable_thinking":false,"messages":[{"role":"user","content":"Write a short story, about 300 words, of a lighthouse keeper who finds a message in a bottle."}]}')
    ST2=$(last_stats)
    check "[10] novel text answers" "$NOVEL" "lighthouse"
    check "[10] novel text stays armed" "$ST2" "runtime_disabled=false"
    case "$ST2" in *"plain=0 "*) echo "FAIL [10] no plain round on novel text (the policy never stepped back)"; fail=$((fail+1));; *) echo "PASS [10] plain rounds carry novel text"; pass=$((pass+1));; esac

    T10=$(chat '{"max_tokens":2000,"temperature":0,"messages":[{"role":"user","content":"What is the weather in Paris and in Tokyo right now? Use the tool."}],"tools":'"$TOOLS"'}')
    check "[10] tool calls under DFlash" "$T10" '"tool_calls"'
    check "[10] Paris call" "$T10" "Paris"
    check "[10] Tokyo call" "$T10" "Tokyo"
    check_absent "[10] no tool markup" "$T10" "<tool_call>"

    DIGITS10=$(chat "$NEEDLE" | python3 -c 'import json,re,sys; print(re.sub(r"\D", "", json.load(sys.stdin)["choices"][0]["message"]["content"]))')
    check "[10] needle recovered at ~8k tokens" "$DIGITS10" "741952"
    B10=$(chat "$NEEDLE")
    CACHED10=$(echo "$B10" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["prompt_tokens_details"]["cached_tokens"])')
    [ "$CACHED10" -gt 0 ] && { echo "PASS [10] prefix cache engaged beside the drafter ($CACHED10 tokens)"; pass=$((pass+1)); } || { echo "FAIL [10] 0 cached tokens"; fail=$((fail+1)); }
    D10=$(echo "$B10" | python3 -c 'import json,re,sys; print(re.sub(r"\D", "", json.load(sys.stdin)["choices"][0]["message"]["content"]))')
    check "[10] the answer survives the cache hit" "$D10" "741952"

    # Company: a burst decodes beside the drafter; whichever slot armed it yields (the log line is the engagement when it did).
    CONC10=$(python3 - "$PORT" <<'PY'
import json, sys, threading, urllib.request
port = sys.argv[1]
reqs = [("Count from 1 to 30, separated by single spaces, and write nothing else.", 30, 150),
        ("Count from 1 to 45, separated by single spaces, and write nothing else.", 45, 220),
        ("Count from 1 to 60, separated by single spaces, and write nothing else.", 60, 300)]
out = {}
def run(i):
    msg, n, mt = reqs[i]
    body = {"max_tokens": mt, "temperature": 0, "enable_thinking": False, "messages": [{"role": "user", "content": msg}]}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    c = json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["message"]["content"].split()
    out[i] = c[:n] == [str(k) for k in range(1, n + 1)]
ts = [threading.Thread(target=run, args=(i,)) for i in range(len(reqs))]
[t.start() for t in ts]; [t.join() for t in ts]
print("count30=%s count45=%s count60=%s" % (out[0], out[1], out[2]))
PY
)
    check "[10] 30-count stream correct beside the drafter" "$CONC10" "count30=True"
    check "[10] 45-count stream correct beside the drafter" "$CONC10" "count45=True"
    check "[10] 60-count stream correct beside the drafter" "$CONC10" "count60=True"
    check_absent "[10] no MLX error with the drafter" "$(cat "$LOG")" "[mlx]"
fi

echo
echo "glm5_next integration: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
