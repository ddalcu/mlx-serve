#!/bin/bash
# Integration test: chat completions emit a llama.cpp-style `timings` block
# alongside `usage`, on both the non-stream JSON body and the final SSE chunk
# of a streaming response: on the usage chunk when stream_options.include_usage
# is set, else on the finish_reason chunk. Either way, exactly once. A streamed
# /v1/responses turn carries the same object at the top level of its
# response.completed event.
#
# Required fields (llama.cpp shape):
#   prompt_n, prompt_ms, prompt_per_second,
#   predicted_n, predicted_ms, predicted_per_second
#
# Run against a live server:
#   ./tests/test_timings.sh [port]

set -u
PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"
FAIL=0

echo "=== Chat-completions timings field ==="
echo "Server: $BASE"

if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo "SKIP: Server not running on port $PORT"
    exit 0
fi

# ── Test 1: non-streaming chat completion ──────────────────────────────────
echo
echo "--- Test 1: non-stream chat.completion carries a top-level 'timings' object ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "messages": [{"role":"user","content":"Say hi in five words."}],
    "max_tokens": 16,
    "temperature": 0.0,
    "stream": false
  }')
if [ -z "$RESULT" ]; then
    echo "  FAIL: empty response from server"
    exit 1
fi

RAW="$RESULT" python3 - <<'PY'
import json, os, sys
r = json.loads(os.environ["RAW"])
t = r.get("timings")
if not isinstance(t, dict):
    print(f"  FAIL: no 'timings' object in response. keys={list(r.keys())}")
    sys.exit(1)
required = ["prompt_n","prompt_ms","prompt_per_second",
           "predicted_n","predicted_ms","predicted_per_second"]
missing = [k for k in required if k not in t]
if missing:
    print(f"  FAIL: timings missing fields: {missing}. got={list(t.keys())}")
    sys.exit(1)
for k in ["prompt_n","predicted_n"]:
    if not isinstance(t[k], int) or t[k] < 0:
        print(f"  FAIL: timings.{k} not a non-negative int: {t[k]!r}")
        sys.exit(1)
for k in ["prompt_ms","predicted_ms","prompt_per_second","predicted_per_second"]:
    if not isinstance(t[k], (int,float)) or t[k] < 0:
        print(f"  FAIL: timings.{k} not a non-negative number: {t[k]!r}")
        sys.exit(1)
# usage must still contain the standard OpenAI counts and agree with timings.
u = r.get("usage", {})
if u.get("prompt_tokens") != t["prompt_n"]:
    print(f"  FAIL: usage.prompt_tokens={u.get('prompt_tokens')} != timings.prompt_n={t['prompt_n']}")
    sys.exit(1)
if u.get("completion_tokens") != t["predicted_n"]:
    print(f"  FAIL: usage.completion_tokens={u.get('completion_tokens')} != timings.predicted_n={t['predicted_n']}")
    sys.exit(1)
# When tokens > 0 the per-second figures must be > 0 (we generated something).
if t["predicted_n"] > 0 and t["predicted_per_second"] <= 0:
    print(f"  FAIL: predicted_n={t['predicted_n']} but predicted_per_second={t['predicted_per_second']}")
    sys.exit(1)
print(f"  PASS  timings={json.dumps(t)}")
PY
RC=$?
[ $RC -ne 0 ] && FAIL=1

# ── Streaming: timings ride exactly one chunk ─────────────────────────────
# stream_check <title> <where> <extra request fields>: where = usage (the
# include_usage chunk, which must agree with timings) or finish (the
# finish_reason chunk, when the client did not ask for usage).
stream_check() {
    echo
    echo "--- $1 ---"
    local stream
    stream=$(curl -sN "$BASE/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "mlx-serve",
        "messages": [{"role":"user","content":"Say hi in five words."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": true'"$3"'
      }')
    if [ -z "$stream" ]; then
        echo "  FAIL: empty stream"
        FAIL=1
        return
    fi
    RAW="$stream" WHERE="$2" python3 - <<'PY'
import json, os, sys
chunks = []
for line in os.environ["RAW"].splitlines():
    if not line.startswith("data: "):
        continue
    payload = line[len("data: "):].strip()
    if payload == "[DONE]" or not payload:
        continue
    try:
        chunks.append(json.loads(payload))
    except json.JSONDecodeError:
        pass

where = os.environ["WHERE"]
if where == "usage":
    hits = [c for c in chunks if isinstance(c.get("usage"), dict) and "prompt_tokens" in c["usage"]]
else:
    hits = [c for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
if not hits:
    print(f"  FAIL: no {where} chunk. chunks={[list(c.keys()) for c in chunks[-3:]]}")
    sys.exit(1)
last = hits[-1]
t = last.get("timings")
if not isinstance(t, dict):
    print(f"  FAIL: {where} chunk has no 'timings'. keys={list(last.keys())}")
    sys.exit(1)
required = ["prompt_n","prompt_ms","prompt_per_second",
           "predicted_n","predicted_ms","predicted_per_second"]
missing = [k for k in required if k not in t]
if missing:
    print(f"  FAIL: stream timings missing fields: {missing}. got={list(t.keys())}")
    sys.exit(1)
if where == "usage":
    u = last["usage"]
    if u.get("prompt_tokens") != t["prompt_n"] or u.get("completion_tokens") != t["predicted_n"]:
        print(f"  FAIL: stream usage/timings token counts disagree: usage={u} timings={t}")
        sys.exit(1)
if t["predicted_n"] > 0 and t["predicted_per_second"] <= 0:
    print(f"  FAIL: predicted_n={t['predicted_n']} but predicted_per_second={t['predicted_per_second']}")
    sys.exit(1)
n = sum(1 for c in chunks if "timings" in c)
if n != 1:
    print(f"  FAIL: timings appears on {n} chunks; expected once, on the {where} chunk")
    sys.exit(1)
print(f"  PASS  timings={json.dumps(t)}")
PY
    [ $? -ne 0 ] && FAIL=1
}

stream_check "Test 2: streaming usage chunk carries 'timings' alongside 'usage'" usage ', "stream_options": {"include_usage": true}'
stream_check "Test 3: without include_usage the finish_reason chunk carries 'timings'" finish ''

# ── Test 4: /v1/responses stream — response.completed carries top-level timings
# The envelope nests timings under `response`; proxies read the top level, so
# the terminal event repeats the same object beside `response`, and no other
# event carries a top-level `timings`.
echo
echo "--- Test 4: Responses stream: response.completed carries top-level 'timings' ---"
RSTREAM=$(curl -sN "$BASE/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-serve",
    "input": "Say hi in five words.",
    "max_output_tokens": 16,
    "temperature": 0.0,
    "stream": true
  }')
if [ -z "$RSTREAM" ]; then
    echo "  FAIL: empty stream"
    FAIL=1
else
    RAW="$RSTREAM" python3 - <<'PY'
import json, os, sys
events = []
for line in os.environ["RAW"].splitlines():
    if not line.startswith("data: "):
        continue
    payload = line[len("data: "):].strip()
    if payload == "[DONE]" or not payload:
        continue
    try:
        events.append(json.loads(payload))
    except json.JSONDecodeError:
        pass
done = [e for e in events if e.get("type") == "response.completed"]
if len(done) != 1:
    print(f"  FAIL: expected one response.completed, got {len(done)}")
    sys.exit(1)
ev = done[0]
t = ev.get("timings")
if not isinstance(t, dict):
    print(f"  FAIL: response.completed has no top-level 'timings'. keys={list(ev.keys())}")
    sys.exit(1)
if t != ev["response"].get("timings"):
    print(f"  FAIL: top-level timings {t} != response.timings {ev['response'].get('timings')}")
    sys.exit(1)
u = ev["response"].get("usage", {})
if u.get("input_tokens") != t.get("prompt_n") or u.get("output_tokens") != t.get("predicted_n"):
    print(f"  FAIL: usage/timings token counts disagree: usage={u} timings={t}")
    sys.exit(1)
if t["predicted_n"] > 0 and t["predicted_per_second"] <= 0:
    print(f"  FAIL: predicted_n={t['predicted_n']} but predicted_per_second={t['predicted_per_second']}")
    sys.exit(1)
n = sum(1 for e in events if "timings" in e)
if n != 1:
    print(f"  FAIL: top-level timings appears on {n} events; expected once, on response.completed")
    sys.exit(1)
print(f"  PASS  timings={json.dumps(t)}")
PY
    [ $? -ne 0 ] && FAIL=1
fi

echo
if [ $FAIL -ne 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL TESTS PASSED"
