#!/usr/bin/env bash
# Plaintext Responses history must reach the same prompt as chat history.
# Usage: REASONING_REPLAY_MODEL=<Qwen model directory> bash tests/test_responses_reasoning.sh [port]
set -euo pipefail
cd "$(dirname "$0")/.."
source tests/lib/server.sh
source tests/_lib_models.sh

MODEL="${REASONING_REPLAY_MODEL:-$(find_model mlx-community/Qwen3.5-0.8B-MLX-4bit || true)}"
[[ -d "$MODEL" ]] || { echo "SKIP: set REASONING_REPLAY_MODEL to a Qwen checkpoint"; exit 0; }
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${1:-11592}"
mkdir -p "$HOME/.cache/mlx-serve-tests"
WORK=$(mktemp -d "$HOME/.cache/mlx-serve-tests/responses-reasoning.XXXXXX")
mkdir -p "$WORK/home"
env HOME="$WORK/home" "$BINARY" --serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    --no-mtp --no-drafter --log-level debug >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
trap 'srv_stop "$SERVER_PID"' EXIT
BASE="http://127.0.0.1:$PORT"
srv_wait_health "$BASE" 120 0.5 "$SERVER_PID" || { tail -20 "$WORK/server.log"; exit 1; }

python3 - "$BASE" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base = sys.argv[1]

def request(path, payload):
    req = urllib.request.Request(base + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            if payload.get("stream"):
                final = None
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                        final = event["response"]
                assert final is not None, "stream has no terminal response"
                return response.status, final
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)

def tokens(path, history, stream=False):
    response_api = path == "/v1/responses"
    payload = {"model": "local", "temperature": 0, "seed": 7}
    if response_api:
        payload.update(input=history, max_output_tokens=8, reasoning={"effort": "none"}, store=False, stream=stream)
    else:
        payload.update(messages=history, max_tokens=8, enable_thinking=False)
    status, body = request(path, payload)
    assert status == 200, (status, body)
    if response_api:
        assert body["status"] in ("completed", "incomplete"), body
    return body["usage"]["input_tokens" if response_api else "prompt_tokens"]

thought = "We need to preserve each step of this calculation. " * 12
user = {"role": "user", "content": "What is 17 plus 25?"}
reasoning = {"type": "reasoning", "summary": [{"type": "summary_text", "text": thought}]}
answer = {"role": "assistant", "content": "The result is 42."}
cases = [
    ([user, reasoning, answer], [user, dict(answer, reasoning_content=thought)], [user, answer]),
    ([user, reasoning], [user, {"role": "assistant", "content": "", "reasoning_content": thought}], [user]),
]
calls = [{"id": f"c{i}", "type": "function", "function": {"name": "lookup", "arguments": "{}"}} for i in (1, 2)]
outputs = [{"type": "function_call_output", "call_id": f"c{i}", "output": str(i)} for i in (2, 1)]
wire_calls = [{"type": "function_call", "call_id": c["id"], **c["function"]} for c in calls]
tool_messages = [{"role": "tool", "tool_call_id": o["call_id"], "content": o["output"]} for o in outputs]
cases.append((
    [user, reasoning, *wire_calls, *outputs],
    [user, {"role": "assistant", "content": "", "reasoning_content": thought, "tool_calls": calls}, *tool_messages],
    [user, *wire_calls, *outputs],
))
for repetition in range(2):
    for index, (wire, chat, stripped) in enumerate(cases):
        actual = tokens("/v1/responses", wire)
        expected = tokens("/v1/chat/completions", chat)
        without = tokens("/v1/responses", stripped)
        assert actual == expected, (index, actual, expected)
        assert actual > without, (index, actual, without)
        streamed = tokens("/v1/responses", wire, stream=True)
        assert streamed == actual, (index, streamed, actual)
        print(f"PASS replay {index}, repeat {repetition}: {actual} prompt tokens, {without} without reasoning")

for bad in (
    {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
    {"type": "reasoning", "summary": "invalid"},
):
    for stream in (False, True):
        assert tokens("/v1/responses", [user, bad], stream=stream) == tokens("/v1/responses", [user]), (bad, stream)
        print(f"PASS unreadable reasoning item dropped: stream={stream}, {bad['summary'] if bad['summary'] == 'invalid' else 'opaque'}")
    status, body = request("/v1/responses/compact", {"model": "local", "input": [user, bad]})
    assert status == 200, (status, body)
    print("PASS unreadable reasoning item dropped: compact")
PY
echo "Server log: $WORK/server.log"
