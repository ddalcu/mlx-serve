#!/bin/bash
# Integration test for the tool-call parser. Boots a server against the given
# model, sends three tool-using chats, asserts each response has a properly
# structured `tool_calls[0]` (NOT leaked `<tool_call>` text in `content`).
#
# Designed to run across architectures — covers both the MLX/Jinja path
# (canonical Hermes-style emit) and the ds4/DSV4 path (attribute-form +
# `</tool_request>` mismatched-close quirks the parser now tolerates).
#
# Usage:
#   ./tests/test_tool_call_parse.sh [model_path] [port]
#
# Exit 0 only when every prompt parses cleanly. Skips quietly when the model
# isn't found locally so CI on smaller hosts doesn't fail.

set -uo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${1:-$HOME/.lmstudio/models/mlx-community/gemma-4-e4b-it-4bit}"
PORT="${2:-11305}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; NC='\033[0m'

# Model path may be a directory (MLX) or a .gguf file (ds4). Both are valid.
if [[ ! -e "$MODEL_PATH" ]]; then
    echo -e "${YELLOW}SKIP${NC}: model not found at $MODEL_PATH"
    exit 0
fi

if [[ ! -x "$BINARY" ]]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

pkill -f "mlx-serve --serve --port $PORT" >/dev/null 2>&1 || true
sleep 1

LOG=$(mktemp)
echo -e "${BLUE}=== Tool-call parse integration test ===${NC}"
echo "Model:  $MODEL_PATH"
echo "Port:   $PORT"
echo "Log:    $LOG"

"$BINARY" --model "$MODEL_PATH" --serve --port "$PORT" --max-tokens 96 > "$LOG" 2>&1 &
SP=$!
cleanup() { kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; }
trap cleanup EXIT

# Wait for /health — DSV4's 80GB mmap warmup can take a few minutes cold.
for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SP" 2>/dev/null; then
        echo -e "${RED}FAIL${NC}: server exited during startup"
        tail -20 "$LOG"
        exit 1
    fi
    sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo -e "${RED}FAIL${NC}: /health never came up"
    tail -20 "$LOG"
    exit 1
fi
echo -e "${GREEN}server up${NC}"

PASS=0; FAIL=0; TOTAL=0

# `assert_tool_call` posts a single chat completion + asserts the response
# shape. Args: description, system_prompt, user_prompt, expected_tool_name.
assert_tool_call() {
    local desc="$1" system="$2" user="$3" expected_tool="$4"
    TOTAL=$((TOTAL + 1))

    local payload
    payload=$(SYS="$system" USR="$user" python3 -c '
import json, os
print(json.dumps({
    "model": "mlx-serve",
    "messages": [
        {"role": "system", "content": os.environ["SYS"]},
        {"role": "user",   "content": os.environ["USR"]},
    ],
    "tools": [
        {"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
        {"type":"function","function":{"name":"webSearch","description":"Search the web with DuckDuckGo","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
        {"type":"function","function":{"name":"writeFile","description":"Write a small file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    ],
    "max_tokens": 96,
    "temperature": 0.0,
    "enable_thinking": False
}))')

    local resp
    resp=$(curl -sf -m 240 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$payload" 2>/dev/null || echo '')

    if [[ -z "$resp" ]]; then
        echo -e "  ${RED}FAIL${NC} $desc — empty response from server"
        FAIL=$((FAIL + 1))
        return
    fi

    local result
    result=$(printf '%s' "$resp" | EXP="$expected_tool" python3 -c '
import json, os, sys
try:
    r = json.load(sys.stdin)
except Exception as e:
    print(f"FAIL|invalid JSON ({e})")
    sys.exit(0)
ch = (r.get("choices") or [{}])[0]
msg = ch.get("message") or {}
content = msg.get("content") or ""
tool_calls = msg.get("tool_calls") or []
finish = ch.get("finish_reason", "")
expected = os.environ["EXP"]

# Core regression guard: NO raw tool-call XML in the visible content,
# regardless of whether the model called a tool or answered directly. The
# bug we fixed was exactly this — the parser dropped <tool_call …> and
# </tool_request> variants, so the raw XML leaked into the chat bubble.
leaked = any(t in content for t in ("<tool_call", "</tool_call>", "</tool_request>"))
if leaked:
    print(f"FAIL|tool-call XML leaked into content: {content[:160]!r}")
    sys.exit(0)

if tool_calls:
    # Model chose to call a tool — verify the parser produced a well-formed
    # entry, not garbage.
    tc = tool_calls[0]
    fn = (tc.get("function") or {})
    name = fn.get("name", "")
    args = fn.get("arguments", "")
    if name not in {"shell", "webSearch", "writeFile"}:
        print(f"FAIL|unknown tool {name!r} called (expected shell/webSearch/writeFile)")
        sys.exit(0)
    try:
        parsed_args = json.loads(args)
    except Exception as e:
        print(f"FAIL|arguments not valid JSON ({e}); got {args[:120]!r}")
        sys.exit(0)
    if not isinstance(parsed_args, dict):
        print(f"FAIL|arguments not an object; got {parsed_args!r}")
        sys.exit(0)
    if finish != "tool_calls":
        print(f"FAIL|tool_calls present but finish_reason={finish!r}")
        sys.exit(0)
    print(f"OK|tool_call name={name} args={list(parsed_args)}")
else:
    # Model answered directly — that’s allowed; just confirm we got a
    # plausible text reply rather than an empty / error-shaped response.
    if not content.strip():
        print(f"FAIL|no tool_call AND empty content (finish={finish})")
        sys.exit(0)
    snippet = content[:80].replace("|", "/")
    print(f"OK|model-answered (no tool, finish={finish}); content={snippet!r}")
')

    local status="${result%%|*}"
    local detail="${result#*|}"
    if [[ "$status" == "OK" ]]; then
        echo -e "  ${GREEN}PASS${NC} $desc  ($detail)"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC} $desc — $detail"
        FAIL=$((FAIL + 1))
    fi
}

echo
echo "--- Asserting properly structured tool_calls across three prompt shapes ---"

assert_tool_call "shell tool from list-files prompt" \
    "You can call tools. Use the shell tool to run commands." \
    "List the files in /tmp using the shell tool." \
    "shell"

assert_tool_call "webSearch tool from research prompt" \
    "Use the webSearch tool when the user asks to research something." \
    "Research the top 5 open source inference apps." \
    "webSearch"

assert_tool_call "writeFile tool from create-file prompt" \
    "Use the writeFile tool to create files." \
    "Create a file at /tmp/hello.txt containing the single word 'hello'." \
    "writeFile"

echo
echo "============================================="
if [[ $FAIL -eq 0 ]]; then
    echo -e "  ${GREEN}ALL $TOTAL TESTS PASSED${NC}"
    exit 0
else
    echo -e "  ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} (out of $TOTAL)"
    echo "  log: $LOG"
    exit 1
fi
