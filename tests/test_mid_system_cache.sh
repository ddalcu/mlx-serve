#!/bin/bash
# Late system notes preserve the conversation prefix across tool rounds and restart.
set -euo pipefail
MODEL="${1:?Usage: test_mid_system_cache.sh <model_dir> [port]}"
PORT="${2:-11368}"
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
CACHE_ENTRIES="${MID_SYSTEM_CACHE_ENTRIES:-2}"
SCRATCH=$(mktemp -d)
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$SCRATCH"
}
trap cleanup EXIT
start_server() {
    HOME="$SCRATCH" "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --ctx-size 16384 --kv-quant 4 --prefix-cache-entries "$CACHE_ENTRIES" --prefix-cache-disk 4GB \
        --ssm-checkpoint-stride 512 --prefill-chunk 512 --no-mtp --no-pld --no-drafter \
        --log-file "$SCRATCH/server.log" > "$SCRATCH/stdout.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do
        if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then return; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            printf 'FAIL: server stopped during load; log: %s\n' "$SCRATCH/stdout.log"
            return 1
        fi
        sleep 1
    done
    return 1
}
stop_server() {
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
check_turns() {
    python3 - "$PORT" "$SCRATCH" "$1" <<'PY'
import json
import pathlib
import sys
import urllib.request

port, scratch, mode = sys.argv[1:]
base = f"http://127.0.0.1:{port}"
state = pathlib.Path(scratch) / "conversation.json"

def post(path, body):
    request = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        if not body.get("stream"):
            return json.load(response)
        usage = {}
        content = []
        stop_reason = None
        stopped = False
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            event = json.loads(raw[6:])
            assert event.get("type") != "error", event
            usage.update(event.get("message", {}).get("usage", {}))
            usage.update(event.get("usage", {}))
            if event.get("delta", {}).get("type") == "text_delta":
                content.append(event["delta"]["text"])
            if event.get("delta", {}).get("stop_reason"):
                stop_reason = event["delta"]["stop_reason"]
            stopped |= event.get("type") == "message_stop"
        assert stopped, "stream did not finish"
        return {"usage": usage, "content": [{"type": "text", "text": "".join(content)}], "stop_reason": stop_reason}

if mode == "restart":
    turns = json.loads(state.read_text())
else:
    text = "\n".join(f"Log {i}: subsystem {i % 17}, value {i * 31 % 997}, checksum {i * i % 9973}." for i in range(180))
    turns = [{"role": "user", "content": text + "\nAcknowledge with OK."},
             {"role": "system", "content": "<total_tokens>15000000 tokens left</total_tokens>"}]

tools = [{"name": "lookup", "description": "Read one log entry", "input_schema": {"type": "object", "properties": {"index": {"type": "integer"}}}}]
for round_index in range(1 if mode == "restart" else 4):
    if mode == "live" and round_index == 3:
        turns.extend([{"role": "assistant", "content": "OK"},
                      {"role": "user", "content": "New user turn: acknowledge again."},
                      {"role": "system", "content": "New turn reminder."}])
    answer = post("/v1/messages", {"model": "mlx-serve", "system": "Read the log and answer briefly.", "messages": turns,
                                    "tools": tools, "max_tokens": 8, "temperature": 0, "stream": mode == "restart" or round_index % 2 == 1,
                                    "thinking": {"type": "enabled", "budget_tokens": 1024}})
    usage = answer["usage"]
    total, cached = usage["input_tokens"], usage.get("cache_read_input_tokens", 0)
    if mode == "restart" or round_index > 0:
        assert cached >= total * 0.85, f"prefix lost: cached={cached}, total={total}"
    print(f"PASS {mode} round {round_index + 1}: cached={cached}/{total}")
    if mode == "restart":
        continue
    turns.extend([
        {"role": "assistant", "content": [{"type": "tool_use", "id": f"call_{round_index}", "name": "lookup", "input": {"index": round_index}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"call_{round_index}", "content": "Entry confirmed."}]},
        {"role": "system", "content": f"<total_tokens>{14999999-round_index} tokens left</total_tokens>"},
    ])
state.write_text(json.dumps(turns))
if mode == "restart":
    body = {"model": "mlx-serve", "system": "Answer briefly.",
        "messages": [{"role": "user", "content": "Name a color."},
                     {"role": "system", "content": "Use one word."}],
        "max_tokens": 32, "temperature": 0, "thinking": {"type": "disabled"}}
    post("/v1/messages", body)
    answers = [post("/v1/messages", dict(body, stream=stream)) for stream in (False, True)]
    texts = ["".join(block.get("text", "") for block in answer["content"] if block.get("type") == "text") for answer in answers]
    assert texts[0] == texts[1], f"stream delivery differs: {texts!r}"
    assert answers[0]["stop_reason"] == answers[1]["stop_reason"], "stop reason differs"
    assert answers[0]["usage"]["output_tokens"] == answers[1]["usage"]["output_tokens"], "output token count differs"
    print("PASS late-system stream/non-stream delivery agrees")
PY
}
start_server
check_turns live
stop_server
start_server
check_turns restart
if grep -q 'jinja render failed' "$SCRATCH/server.log"; then
    printf 'FAIL: native template fell back\n'
    exit 1
fi
if [ "$CACHE_ENTRIES" = 0 ] && grep -q '\[hot-cache\] resident=' "$SCRATCH/server.log"; then
    printf 'FAIL: idle RAM cache retained\n'
    exit 1
fi
printf 'PASS: late system notes, tool history, restart (RAM entries=%s)\n' "$CACHE_ENTRIES"
