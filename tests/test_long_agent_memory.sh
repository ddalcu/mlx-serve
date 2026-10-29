#!/bin/bash
# test_long_agent_memory.sh — an 11-turn Claude-Code-style agent conversation keeps its memory: three facts planted in
# turn 1 (codename, language, deadline) are recalled across mode transitions (tools on/off, thinking on/off, mocked
# tools, a history past the sliding window), and the final summary names all three.
#
#   ./tests/test_long_agent_memory.sh [port]
#
# Uses a server already on the port, else starts E4B + drafter (the Claude Code launch shape). A failed recall is
# the "acts like it is the first time" class: thinking leaking back as input, a dropped turn, a bad cache reset.

set -u

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"
MODEL="${LONG_AGENT_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}"
DRAFTER="${LONG_AGENT_TEST_DRAFTER:-$HOME/.mlx-serve/models/mlx-community/gemma-4-E4B-it-assistant-bf16}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_long_agent_memory: model not found ($MODEL)"
    exit 0
fi

# Reuse an already-running server if present, otherwise start our own.
OWN_SERVER=0
LOG=""
if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    if [ ! -x ./zig-out/bin/mlx-serve ]; then
        echo -e "${RED}FAIL${NC} mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
        exit 1
    fi
    LOG=$(mktemp)
    DRAFTER_ARGS=()
    if [ -d "$DRAFTER" ]; then
        DRAFTER_ARGS=(--drafter "$DRAFTER")
    fi
    echo "Starting mlx-serve on port $PORT (log: $LOG)..."
    ./zig-out/bin/mlx-serve --model "$MODEL" --serve --port "$PORT" --log-level info \
        ${DRAFTER_ARGS[@]+"${DRAFTER_ARGS[@]}"} ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOG" 2>&1 &
    SERVER_PID=$!
    OWN_SERVER=1
    trap 'if [ "$OWN_SERVER" = "1" ]; then kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; fi' EXIT

    for _ in $(seq 1 90); do
        if curl -sf "$BASE/health" > /dev/null 2>&1; then break; fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo -e "${RED}FAIL${NC} server crashed during load"
            tail -30 "$LOG"
            exit 1
        fi
        sleep 1
    done
fi

# Drive the whole test from python so we can keep one growing message list and
# do clean SSE/JSON parsing without bash quoting nightmares.
# Not exec: bash has to outlive python for its EXIT trap to stop the server.
python3 "$(dirname "$0")/test_long_agent_memory.py" "$PORT"
exit $?
