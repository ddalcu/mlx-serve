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
#   [5] launch claude --print: env-only script, no config file, ADVERTISED
#       context declared verbatim (CLAUDE_CODE_MAX_CONTEXT_TOKENS — without it
#       Claude Code assumes 200k and auto-compacts there) + the derived output
#       budget
#   [6] extra args after -- ride the agent invocation line
#   [7] launch opencode2 --print (compatibility alias): XDG_CONFIG_HOME under
#       the dedicated dir, standalone invocation of the RESOLVED binary
#       (v2 `opencode` when installed, else legacy `opencode2`), cli.json
#       carries metricsUrl = base + /metrics.json, plugin has tui.tsx
#   [8] launch opencode --print detects `opencode --version` and routes: a
#       2.x install gets the v2 arm (standalone + model pinned in the config)
#       and names the detected version on stderr
#   [9] launch zcode --print: script exports the dedicated data dir + provider
#       config file and invokes zcode; provider_config.json targets base + /v1
#       with the ADVERTISED context for the served model
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
mkdir -p "${TMPDIR:?set TMPDIR to a workspace scratch directory}"
"$BIN" --model "$MODEL_DIR" --serve --port "$PORT" >"$TMPDIR/mlx-serve-launch-test.log" 2>&1 &
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
EXPECT_OUT=$(python3 -c "print(min(65536, max(1024, $ADV_CTX // 2)))")
OK=1
echo "$OUT" | grep -q "export ANTHROPIC_BASE_URL='$BASE'" || OK=0
echo "$OUT" | grep -q "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=$EXPECT_OUT" || OK=0
# Without this, Claude Code assumes 200k for an off-catalog model and
# auto-compacts there — a 786k server driven as a 200k one.
echo "$OUT" | grep -q "export CLAUDE_CODE_MAX_CONTEXT_TOKENS=$ADV_CTX" || OK=0
echo "$OUT" | grep -qF 'claude --plugin-dir "$HOME/.mlx-serve/claude/plugin" --model '"$MODEL_ID" || OK=0
if [ "$OK" = 1 ]; then
    run_test "claude script is env-only with the advertised context + derived output budget" PASS
else
    run_test "claude script is env-only with the advertised context + derived output budget" FAIL "$OUT"
fi

# ── [6] passthrough args ──
OUT=$("$BIN" launch codex --print --url "$BASE" -- resume 2>&1)
if echo "$OUT" | grep -q "\"\$CODEX_BIN\" 'resume'"; then
    run_test "extra args after -- ride the agent invocation" PASS
else
    run_test "extra args after -- ride the agent invocation" FAIL "$OUT"
fi

# ── [7] launch opencode2 --print (compatibility alias) ──
# The alias forces the v2 profile and resolves the binary the same way:
# `opencode` when it is major >= 2, else a legacy `opencode2` on PATH.
OC_VER=$(opencode --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+[0-9.]*' | head -1)
OC_MAJOR=${OC_VER%%.*}
EXPECTED_BIN=""
if [ "${OC_MAJOR:-0}" -ge 2 ] 2>/dev/null; then EXPECTED_BIN=opencode; fi
if [ -z "$EXPECTED_BIN" ] && command -v opencode2 >/dev/null 2>&1; then EXPECTED_BIN=opencode2; fi
if [ -z "$EXPECTED_BIN" ]; then
    echo "  SKIP: no OpenCode v2 binary on PATH for [7]/[8]"
else
OUT=$("$BIN" launch opencode2 --print --url "$BASE" 2>&1)
OK=1
echo "$OUT" | grep -q 'export XDG_CONFIG_HOME="$HOME/.mlx-serve/opencode2"' || OK=0
echo "$OUT" | grep -q 'export OPENCODE_CONFIG_CONTENT=' || OK=0
echo "$OUT" | grep -q "^$EXPECTED_BIN --standalone\$" || OK=0
echo "$OUT" | grep -q "\"model\": \"mlx/$MODEL_ID\"" || OK=0
CLI_JSON="$HOME/.mlx-serve/opencode2/opencode/cli.json"
if [ ! -f "$CLI_JSON" ]; then
    OK=0
else
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
want = sys.argv[2] + '/metrics.json'
plugins = d.get('plugins') or []
# a user's own plugins ride through the merge as plain strings
mlx = [p for p in plugins if isinstance(p, dict) and (p.get('package') or '').endswith('mlx-serve')]
assert len(mlx) == 1, mlx
assert mlx[0].get('options', {}).get('metricsUrl') == want, mlx[0]
assert 'metricsToken' not in (mlx[0].get('options') or {})
" "$CLI_JSON" "$BASE" || OK=0
fi
[ -f "$HOME/.mlx-serve/opencode2/opencode/plugins/mlx-serve/tui.tsx" ] || OK=0
if [ "$OK" = 1 ]; then
    run_test "opencode2 alias: v2 script via the resolved binary + cli.json + plugin tui.tsx" PASS
else
    run_test "opencode2 alias: v2 script via the resolved binary + cli.json + plugin tui.tsx" FAIL "$OUT"
fi

# ── [8] launch opencode --print routes on the detected version ──
OUT=$("$BIN" launch opencode --print --url "$BASE" 2>&1)
OK=1
if [ "${OC_MAJOR:-0}" -ge 2 ] 2>/dev/null; then
    # A v2 install under the canonical name: standalone, no --model flag,
    # the detection names the real version and the profile it picked.
    echo "$OUT" | grep -q '^opencode --standalone$' || OK=0
    echo "$OUT" | grep -q -- '--model mlx/' && OK=0
    echo "$OUT" | grep -q "using the v2 integration" || OK=0
else
    echo "$OUT" | grep -q '^opencode --model mlx/' || OK=0
    echo "$OUT" | grep -q "using the v1 integration" || OK=0
fi
if [ "$OK" = 1 ]; then
    run_test "launch opencode --print routes the detected version" PASS
else
    run_test "launch opencode --print routes the detected version" FAIL "$OUT"
fi
fi

# ── [9] zcode --print ──
OUT=$("$BIN" launch zcode --print --url "$BASE" 2>&1)
OK=1
echo "$OUT" | grep -q 'export ZCODE_DATA_BASE_DIR="$HOME/.mlx-serve/zcode"' || OK=0
echo "$OUT" | grep -q 'export ZCODE_PERSONAL_PROVIDER_CONFIG_FILE="$HOME/.mlx-serve/zcode/provider_config.json"' || OK=0
echo "$OUT" | grep -q '^zcode$' || OK=0
python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    c = json.load(f)['config']
base, model, ctx = sys.argv[2], sys.argv[3], int(sys.argv[4])
assert c['providerConfigRules']['providerRules'][0]['config']['api']['baseUrl'] == base + '/v1', c
assert c['defaultModelSelection']['modelId'] == model, c
rule = [r for r in c['modelConfigRules']['providerModelRules'] if r['modelId'] == model]
assert rule and rule[0]['config']['properties']['contextWindow'] == ctx, rule
" "$HOME/.mlx-serve/zcode/provider_config.json" "$BASE" "$MODEL_ID" "$ADV_CTX" || OK=0
if [ "$OK" = 1 ]; then
    run_test "zcode script + provider_config.json carry the advertised context" PASS
else
    run_test "zcode script + provider_config.json carry the advertised context" FAIL "$OUT"
fi

echo ""
echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
