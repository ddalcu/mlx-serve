#!/bin/bash
# Integration test: `mlx-serve launch <agent>` (issue #188) — configures and
# launches a third-party coding agent against the local server, ollama-style.
#
# Pins:
#   [1] unknown agent → error naming the choices
#   [2] no server + --no-start → instructions to start one, exit 1
#   [3] launch omp --print against a live server: script exports the pi-spelled
#       agent dir var, targets the served model, and the written models.yml
#       carries the server's ADVERTISED context (never a hardcoded one)
#   [4] launch codex --print: config.toml targets our /v1/responses
#       (wire_api = "responses") with the advertised context
#   [5] launch claude --print: env-only script, no config file, output budget
#       derived from the advertised context
#   [6] extra args after -- ride the agent invocation line
#
# The configs land in the same dedicated ~/.mlx-serve/<agent>/ dirs the app's
# launcher writes (never a user's real agent config) — asserted per agent.

set -u

MODEL_DIR=${1:-~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
PORT=${2:-8097}
BASE="http://127.0.0.1:$PORT"
BIN=./zig-out/bin/mlx-serve
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi
if [ ! -x "$BIN" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

run_test() {
    TOTAL=$((TOTAL+1))
    if [ "$2" = PASS ]; then PASS=$((PASS+1)); echo "  PASS: $1"
    else FAIL=$((FAIL+1)); echo "  FAIL: $1 — $3"; fi
}

# ── [1] unknown agent ──
OUT=$("$BIN" launch not-an-agent 2>&1)
if [ $? -ne 0 ] && echo "$OUT" | grep -q "claude" && echo "$OUT" | grep -q "aider"; then
    run_test "unknown agent errors naming the choices" PASS
else
    run_test "unknown agent errors naming the choices" FAIL "$OUT"
fi

# ── [2] no server, --no-start ──
OUT=$("$BIN" launch omp --no-start --url http://127.0.0.1:59999 2>&1)
if [ $? -ne 0 ] && echo "$OUT" | grep -qi "mlx-serve serve"; then
    run_test "dead server + --no-start instructs how to start one" PASS
else
    run_test "dead server + --no-start instructs how to start one" FAIL "$OUT"
fi

echo "Starting server..."
"$BIN" --model "$MODEL_DIR" --serve --port "$PORT" >/tmp/mlx-serve-launch-test.log 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 40); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    [ "$i" -eq 40 ] && { echo "FAIL: server did not start"; exit 1; }
    sleep 1
done

MODEL_ID=$(basename "$MODEL_DIR")
ADV_CTX=$(curl -s "$BASE/v1/models" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
print((r["data"][0].get("meta") or {}).get("context_length") or 0)
')

# ── [3] omp --print ──
OUT=$("$BIN" launch omp --print --url "$BASE" 2>&1)
OK=1
echo "$OUT" | grep -q 'export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"' || OK=0
echo "$OUT" | grep -q "omp --model mlx/$MODEL_ID" || OK=0
grep -q "contextWindow: $ADV_CTX" ~/.mlx-serve/omp/models.yml || OK=0
grep -q "baseUrl: $BASE/v1" ~/.mlx-serve/omp/models.yml || OK=0
if [ "$OK" = 1 ]; then
    run_test "omp script + models.yml carry the advertised context" PASS
else
    run_test "omp script + models.yml carry the advertised context" FAIL "$OUT"
fi

# ── [4] codex --print ──
OUT=$("$BIN" launch codex --print --url "$BASE" 2>&1)
OK=1
echo "$OUT" | grep -q 'export CODEX_HOME="$HOME/.mlx-serve/codex"' || OK=0
# desktop-app fallback: the ChatGPT/Codex app bundles the CLI off PATH
echo "$OUT" | grep -q '/Applications/ChatGPT.app' || OK=0
echo "$OUT" | grep -q 'Contents/Resources/codex' || OK=0
grep -q 'wire_api = "responses"' ~/.mlx-serve/codex/config.toml || OK=0
grep -q "model_context_window = $ADV_CTX" ~/.mlx-serve/codex/config.toml || OK=0
grep -q "base_url = \"$BASE/v1\"" ~/.mlx-serve/codex/config.toml || OK=0
if [ "$OK" = 1 ]; then
    run_test "codex config targets /v1/responses with the advertised context" PASS
else
    run_test "codex config targets /v1/responses with the advertised context" FAIL "$OUT"
fi

# ── [5] claude --print ──
OUT=$("$BIN" launch claude --print --url "$BASE" 2>&1)
EXPECT_OUT=$(python3 -c "print(min(65536, max(1024, $ADV_CTX // 4)))")
OK=1
echo "$OUT" | grep -q "export ANTHROPIC_BASE_URL='$BASE'" || OK=0
echo "$OUT" | grep -q "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=$EXPECT_OUT" || OK=0
echo "$OUT" | grep -q "claude --model $MODEL_ID" || OK=0
if [ "$OK" = 1 ]; then
    run_test "claude script is env-only with the derived output budget" PASS
else
    run_test "claude script is env-only with the derived output budget" FAIL "$OUT"
fi

# ── [6] passthrough args ──
OUT=$("$BIN" launch codex --print --url "$BASE" -- resume 2>&1)
if echo "$OUT" | grep -q "\"\$CODEX_BIN\" 'resume'"; then
    run_test "extra args after -- ride the agent invocation" PASS
else
    run_test "extra args after -- ride the agent invocation" FAIL "$OUT"
fi

echo ""
echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
