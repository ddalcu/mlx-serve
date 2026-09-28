#!/bin/bash
# --tool-call-stream over HTTP on a Qwen3.5 pack (the `<function=NAME>` dialect).
#   early (opt-in): the first tool_calls delta is id + name with empty arguments
#     and lands in the first half of the request; the arguments follow once, keyed
#     by index only; the assembled calls equal the non-streaming ones. /v1/messages
#     starts the tool_use block that early, with a valid block lifecycle. Ollama
#     /api/chat still receives whole calls with object arguments.
#   A repetition-loop cut after the header closes it with {} arguments and keeps
#     finish_reason stop + finish_details (the decision the gotcha records).
#   end (the default, no flag): one tool_calls delta per call, id + name + arguments together.
# A turn the model answers without a tool SKIPs its checks (model choice, not a bug).
#
# Usage: ./tests/test_tool_call_stream.sh [model_dir] [port]   (TOOL_STREAM_MODEL=<dir>)
# Starts its own servers; no spec decode and no prefix cache, so greedy stream and
# non-stream produce the same tokens on a hybrid Qwen3.5 trunk.

set -u
source "$(dirname "$0")/_lib_models.sh"

MODEL="${1:-${TOOL_STREAM_MODEL:-$(find_model mlx-community/Qwen3.5-4B-MLX-4bit mlx-community/Qwen3.5-0.8B-MLX-4bit)}}"
PORT="${2:-11274}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
LOG=/tmp/test_tool_call_stream.log
PASS=0
FAIL=0
SKIP=0
SERVER_PID=""

[ -n "$MODEL" ] && [ -d "$MODEL" ] || { echo "SKIP: no Qwen3.5 pack found (set TOOL_STREAM_MODEL)"; exit 0; }
[ -x "$BINARY" ] || { echo "FAIL: $BINARY not built"; exit 1; }

start_server() {
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null && wait $SERVER_PID 2>/dev/null
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    sleep 1
    "$BINARY" --model "$MODEL" --serve --port "$PORT" --ctx-size 8192 --no-mtp --no-drafter --prefix-cache-entries 0 "$@" > "$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do
        curl -sf "$BASE/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    echo "FAIL: server did not come up"; tail -20 "$LOG"; exit 1
}

PROBE=$(mktemp)
trap '[ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null; rm -f "$PROBE"' EXIT
cat > "$PROBE" <<'PY'
import json, sys, time, urllib.request

base, mode = sys.argv[1], sys.argv[2]
TOOL = {"name": "write_file", "description": "Write text to a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}
ASK = "Use the write_file tool to save a 20-line poem about the sea to /tmp/sea.txt."


def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def sse(path, body):
    """(seconds since the request, event) for every SSE data line."""
    t0, out = time.monotonic(), []
    with post(path, body) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                out.append((time.monotonic() - t0, json.loads(line[5:])))
    return out


def say(ok, desc):
    print(("PASS " if ok else "FAIL ") + desc)


def chat_body(stream):
    return {"model": "m", "stream": stream, "temperature": 0, "seed": 7, "max_tokens": 1200, "enable_thinking": False,
            "messages": [{"role": "user", "content": ASK}], "tools": [{"type": "function", "function": TOOL}]}


if mode in ("chat-early", "chat-end"):
    deltas = [(t, tc) for t, ev in sse("/v1/chat/completions", chat_body(True))
              for ch in ev.get("choices") or [] for tc in (ch.get("delta") or {}).get("tool_calls") or []]
    if not deltas:
        print("SKIP model answered without a tool")
        sys.exit()
    calls = {}
    for _, tc in deltas:
        c = calls.setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
        fn = tc.get("function") or {}
        c["id"] = c["id"] or tc.get("id", "")
        c["name"] = c["name"] or fn.get("name", "")
        c["arguments"] += fn.get("arguments", "")
    if mode == "chat-end":
        first = deltas[0][1]
        say(len(deltas) == len(calls), "one tool_calls delta per call")
        say(bool(first.get("id") and first["function"].get("name") and first["function"].get("arguments")),
            "that delta carries id, name and arguments")
    else:
        zero = [(t, tc) for t, tc in deltas if tc["index"] == 0]
        (t_head, head), (t_args, args) = zero[0], zero[-1]
        say(bool(head.get("id") and head["function"].get("name")) and head["function"].get("arguments") == "",
            "first delta is the header: id + name, empty arguments")
        say(len(zero) == 2 and "id" not in args and "name" not in args["function"],
            "the arguments follow once, keyed by index only")
        if len(calls[0]["arguments"]) >= 400:
            say(t_args - t_head > 0.5 * t_args, f"header at {t_head:.2f}s, arguments at {t_args:.2f}s")
        else:
            print(f"SKIP timing: arguments only {len(calls[0]['arguments'])} chars")
    with post("/v1/chat/completions", chat_body(False)) as r:
        msg = json.load(r)["choices"][0]["message"]
    whole = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg.get("tool_calls") or []]
    streamed = [(c["name"], json.loads(c["arguments"])) for _, c in sorted(calls.items())]
    say(whole == streamed, "assembled stream calls equal the non-streaming calls")

elif mode == "chat-loop":
    body = chat_body(True)
    body["max_tokens"] = 3000
    body["messages"] = [{"role": "user", "content": "Use the write_file tool to save /tmp/loop.txt. Its content is the line "
                         "'ping pong ping pong' repeated hundreds of times, with no ending."}]
    deltas, finish, details = [], None, {}
    for _, ev in sse("/v1/chat/completions", body):
        for ch in ev.get("choices") or []:
            deltas += (ch.get("delta") or {}).get("tool_calls") or []
            if ch.get("finish_reason"):
                finish, details = ch["finish_reason"], ch.get("finish_details") or {}
    if not deltas or details.get("type") != "repetition_loop":
        print("SKIP no header, or the loop guard did not cut this run")
        sys.exit()
    args = "".join((tc.get("function") or {}).get("arguments", "") for tc in deltas if tc["index"] == 0)
    say(finish == "stop" and args == "{}", f"a loop cut closes the header with {{}} and keeps stop (got {finish}, {args[:40]!r})")

elif mode == "messages":
    body = {"model": "m", "stream": True, "temperature": 0, "max_tokens": 1200, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": ASK}],
            "tools": [{"name": TOOL["name"], "description": TOOL["description"], "input_schema": TOOL["parameters"]}]}
    open_blocks, err, tool_index, t_start, t_input, tool_input = {}, None, None, None, None, ""
    for t, ev in sse("/v1/messages", body):
        typ, idx = ev.get("type"), ev.get("index")
        if typ == "content_block_start":
            if idx in open_blocks:
                err = err or f"index {idx} started twice"
            open_blocks[idx] = ev["content_block"]["type"]
            if ev["content_block"]["type"] == "tool_use" and tool_index is None:
                tool_index, t_start = idx, t
        elif typ == "content_block_delta":
            if idx not in open_blocks:
                err = err or f"delta for unopened index {idx}"
            if idx == tool_index and ev["delta"].get("type") == "input_json_delta":
                t_input, tool_input = t, tool_input + ev["delta"]["partial_json"]
        elif typ == "content_block_stop":
            if open_blocks.pop(idx, None) is None:
                err = err or f"stop for unopened index {idx}"
        elif typ == "message_stop" and open_blocks:
            err = err or f"blocks open at message_stop: {sorted(open_blocks)}"
    if tool_index is None:
        print("SKIP model answered without a tool")
        sys.exit()
    say(err is None, f"content-block lifecycle is valid ({err or 'ok'})")
    say(isinstance(json.loads(tool_input or "null"), dict), "tool input is one JSON object")
    if len(tool_input) >= 400:
        say(t_input - t_start > 0.5 * t_input, f"tool_use start at {t_start:.2f}s, input at {t_input:.2f}s")
    else:
        print(f"SKIP timing: input only {len(tool_input)} chars")

elif mode == "ollama":
    body = {"model": "m", "stream": True, "think": False, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": ASK}], "tools": [{"type": "function", "function": TOOL}]}
    calls = []
    with post("/api/chat", body) as r:
        for raw in r:
            if raw.strip():
                calls += (json.loads(raw).get("message") or {}).get("tool_calls") or []
    if not calls:
        print("SKIP model answered without a tool")
        sys.exit()
    fns = [c["function"] for c in calls]
    say(all(f["name"] and isinstance(f["arguments"], dict) and "raw" not in f["arguments"] for f in fns)
        and "path" in fns[0]["arguments"], "whole calls with object arguments")
PY

run() { # title mode
    echo "$1"
    local out rc
    out=$(python3 "$PROBE" "$BASE" "$2" 2>&1)
    rc=$?
    while IFS= read -r line; do
        case "$line" in
            PASS*) PASS=$((PASS + 1)); echo "  PASS ${line#PASS }" ;;
            FAIL*) FAIL=$((FAIL + 1)); echo "  FAIL ${line#FAIL }" ;;
            SKIP*) SKIP=$((SKIP + 1)); echo "  SKIP ${line#SKIP }" ;;
            *) echo "    $line" ;;
        esac
    done <<< "$out"
    [ $rc -eq 0 ] || { FAIL=$((FAIL + 1)); echo "  FAIL probe exited $rc"; }
}

echo "0. an unknown --tool-call-stream value is refused by name"
"$BINARY" --model /nonexistent --tool-call-stream sometimes > "$LOG" 2>&1
if [ $? -ne 0 ] && grep -q -- "--tool-call-stream" "$LOG"; then
    PASS=$((PASS + 1)); echo "  PASS refused"
else
    FAIL=$((FAIL + 1)); echo "  FAIL not refused by name"
fi

start_server --tool-call-stream early
run "1. early: /v1/chat/completions" chat-early
run "2. early: /v1/messages" messages
run "3. early: Ollama /api/chat keeps whole calls" ollama
run "4. early: a repetition-loop cut after the header" chat-loop

start_server
run "5. end (default, no flag): /v1/chat/completions" chat-end

start_server --tool-call-stream end
run "6. end (explicit): /v1/chat/completions" chat-end

echo ""
echo "===== $PASS passed, $FAIL failed, $SKIP skipped ====="
[ "$FAIL" -eq 0 ] || exit 1
