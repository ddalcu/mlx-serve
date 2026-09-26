#!/bin/bash
# A client stop string ends the ANSWER, never the reasoning (#549): a match
# inside the think block must not cut it. `stop: ["Paris"]` on a thinking model,
# every text surface, stream and non-stream (+ the tools-gated chat stream).
# Bar per case: the reasoning (if the model thought) still contains the stop and the
# answer is cut before it; at least one case must have reasoned past the stop.
#
#   ./tests/test_stop_in_reasoning.sh [model_dir] [port]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
PORT="${2:-11549}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
[[ -d "$MODEL" ]] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[[ -x "$BINARY" ]] || { echo "SKIP: $BINARY missing"; exit 0; }

WORK=$(mktemp -d "$HOME/claude-tmp/stop-reasoning.XXXXXX")
mkdir -p "$WORK/home"
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 0.5
HOME="$WORK/home" "$BINARY" --serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    --log-level debug > "$WORK/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT
for _ in $(seq 1 90); do curl -sf -m 2 "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf -m 2 "$BASE/health" >/dev/null || { echo "FAIL: server did not start"; tail -20 "$WORK/server.log"; exit 1; }

BASE="$BASE" python3 - <<'PY'
import json, os, sys, urllib.request

BASE = os.environ["BASE"]
STOP = "Paris"
PROMPT = "Think about the question, then answer in one sentence: what is the capital of France?"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)

def sse(resp):
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[6:])

def chat(stream, tools=False):
    body = {"model": "m", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 600,
            "temperature": 0, "stop": [STOP], "stream": stream, "reasoning_budget_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": True}}
    if tools: body["tools"] = TOOLS
    r = post("/v1/chat/completions", body)
    if not stream:
        m = json.load(r)["choices"][0]["message"]
        return m.get("reasoning_content") or "", m.get("content") or "", bool(m.get("tool_calls"))
    rc, c, tc = "", "", False
    for ev in sse(r):
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            rc += d.get("reasoning_content") or ""
            c += d.get("content") or ""
            tc |= bool(d.get("tool_calls"))
    return rc, c, tc

def messages(stream):
    body = {"model": "m", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 600,
            "temperature": 0, "stop_sequences": [STOP], "stream": stream,
            "thinking": {"type": "enabled", "budget_tokens": 200}}
    r = post("/v1/messages", body)
    if not stream:
        blocks = json.load(r)["content"]
        return ("".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking"),
                "".join(b.get("text", "") for b in blocks if b["type"] == "text"), False)
    rc, c = "", ""
    for ev in sse(r):
        d = ev.get("delta", {})
        rc += d.get("thinking") or ""
        c += d.get("text") or ""
    return rc, c, False

def responses(stream):
    body = {"model": "m", "input": PROMPT, "max_output_tokens": 600, "temperature": 0,
            "stop": [STOP], "stream": stream, "reasoning": {"effort": "high"}, "reasoning_budget_tokens": 200}
    r = post("/v1/responses", body)
    if not stream:
        out = json.load(r)["output"]
        rc = "".join(p.get("text", "") for o in out if o["type"] == "reasoning" for p in o.get("content", []) + o.get("summary", []))
        c = "".join(p.get("text", "") for o in out if o["type"] == "message" for p in o.get("content", []))
        return rc, c, False
    rc, c = "", ""
    for ev in sse(r):
        t = ev.get("type", "")
        if t.startswith("response.reasoning") and t.endswith(".delta"): rc += ev.get("delta", "")
        elif t == "response.output_text.delta": c += ev.get("delta", "")
    return rc, c, False

fails = 0
thought = 0
for name, fn in [("chat non-stream", lambda: chat(False)), ("chat stream", lambda: chat(True)),
                 ("chat stream + tools", lambda: chat(True, True)),
                 ("messages non-stream", lambda: messages(False)), ("messages stream", lambda: messages(True)),
                 ("responses non-stream", lambda: responses(False)), ("responses stream", lambda: responses(True))]:
    rc, c, tc = fn()
    # A case where the model did not think only checks the answer cut.
    ok = (STOP in rc or rc == "") and c.strip() != "" and STOP not in c
    fails += not ok
    thought += STOP in rc
    print(f"  {'PASS' if ok else 'FAIL'} {name}: reasoning={len(rc)}b has_stop={STOP in rc} content={c[:60]!r}{' tool_call' if tc else ''}")
if not thought:
    print("  FAIL no case reasoned past the stop: nothing proves the thought was spared")
    fails += 1
print(f"{7 - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
PY
