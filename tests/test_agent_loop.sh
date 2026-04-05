#!/bin/bash
# Integration test: multi-turn agent loop with streaming, thinking, and mock tools.
#
# Simulates the full agent workflow:
#   1. User asks to find top news sites, grab headlines, write an HTML report
#   2. Model calls webSearch → mock results
#   3. Model calls browse (multiple sites) → mock headlines
#   4. Model calls writeFile → verify HTML output
#
# Tests: streaming SSE parsing, thinking/reasoning, tool call detection,
#        multi-turn tool result feeding, and final content generation.
#
# Usage: ./tests/test_agent_loop.sh [port]
# Requires a running server with a model loaded.

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"

echo "=== Agent Loop Integration Test ==="
echo "Server: $BASE"

if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo "SKIP: Server not running on port $PORT"
    exit 0
fi

exec python3 "$(dirname "$0")/test_agent_loop.py" "$PORT"
