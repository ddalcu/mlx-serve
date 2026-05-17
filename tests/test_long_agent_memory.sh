#!/bin/bash
# Long-running Claude-Code-style agent memory test.
#
# Existing coverage:
#   - test_long_conversation_spec.sh — 12 well-formedness turns (no memory check)
#   - test_agent_loop.sh             — 6 scenarios, each one-shot (no recall)
#   - test_kv_cache_*                — single-prompt timing + poison checks
#
# What this catches that they don't:
#   - "Model acts like the first time it's seen the task" on turn N: an
#     anti-pattern where thinking from turn N-1 leaks back as input, or KV
#     cache truncation drops earlier turns, or the chat template fails to
#     separate prior assistant turns from the current request — and the model
#     answers as if the planted facts from turn 1 were never there.
#
#   - Cache-invalidation races across mode transitions in a long
#     conversation: tools-on → tools-off → thinking-on → tools-on with the
#     same growing history, where the wrong reset poisons turn N+1's view of
#     turn 1's facts.
#
#   - Thinking + tools + long context together: the existing thinking_tools
#     test is single-turn; this exercises the same mode across an 11-turn
#     conversation that grows past the model's sliding window so we catch
#     state corruption that only surfaces deep in.
#
# Test design:
#   1. Plant three orthogonal facts in turn 1 (codename, language, deadline).
#   2. Drive 10 more turns mixing: tool calls (mocked, no network), thinking
#      passes, plain user turns, recall checks. Tools are mocked locally so
#      the test is fully offline and deterministic.
#   3. After every recall turn, assert the planted fact is in the response.
#      A failure means memory was lost — exactly the user-reported bug.
#   4. Final summary turn must surface ALL three planted facts, proving the
#      whole conversation is still in context after >10 turns of growth.
#
# Usage:
#   ./tests/test_long_agent_memory.sh [port]
#
# If a server is already up on $port, we use it. Otherwise we start one with
# E4B + drafter (the default Claude Code launch shape) so the test exercises
# the spec-decode path on every turn.

set -u

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"
MODEL="${LONG_AGENT_TEST_MODEL:-$HOME/.mlx-serve/models/gemma-4-e4b-it-8bit}"
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
        "${DRAFTER_ARGS[@]}" ${MLX_SERVE_TEST_EXTRA_ARGS:-} > "$LOG" 2>&1 &
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
exec python3 "$(dirname "$0")/test_long_agent_memory.py" "$PORT"
