#!/bin/bash
# GET /launch: the web console's Code Launcher. The page shows
# `curl -fsSL <server>/launch?agent=<a>&model=<id> | sh`; the script must write
# the agent's configs on the machine that runs it and start the agent against
# the host the request reached.
#
# FULLY HERMETIC: the models are stub dirs holding a config.json (nothing
# loads), the agents and the login shell are fakes on PATH, HOME is scratch.
#
# Usage: ./tests/test_launch_script.sh [port]

set -u

PORT=${1:-8133}
BASE="http://127.0.0.1:$PORT"
BIN=./zig-out/bin/mlx-serve
PASS=0
FAIL=0
TOTAL=0

[ -x "$BIN" ] || { echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"; exit 1; }

T=$(mktemp -d)
mkdir -p "$T/root/org/chat-a" "$T/root/org/chat-b" "$T/home" "$T/bin" "$T/server-home"
printf '{"model_type":"qwen3"}' > "$T/root/org/chat-a/config.json"
printf '{"model_type":"qwen3"}' > "$T/root/org/chat-b/config.json"

# A login shell that keeps PATH (a real zsh -l reorders it), and agents that
# record how they were started.
cat > "$T/bin/zsh" <<'EOF'
#!/bin/sh
[ "$1" = -l ] && shift
[ "$1" = -c ] && shift
exec /bin/sh -c "$1"
EOF
cat > "$T/bin/pi" <<'EOF'
#!/bin/sh
echo "dir=$PI_CODING_AGENT_DIR url=$MLX_SERVE_URL args=$*" > "$HOME/agent.out"
EOF
cat > "$T/bin/opencode" <<'EOF'
#!/bin/sh
[ "$1" = --version ] && { echo "2.1.0"; exit 0; }
echo "xdg=$XDG_CONFIG_HOME args=$*" > "$HOME/agent.out"
EOF
chmod +x "$T/bin/"*

HOME="$T/server-home" "$BIN" serve --host 127.0.0.1 --port "$PORT" --model-dir "$T/root" --log-file off >"$T/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
for i in $(seq 1 30); do
    curl -sf "$BASE/health" >/dev/null 2>&1 && break
    [ "$i" -eq 30 ] && { echo "FAIL: server did not start"; cat "$T/server.log"; exit 1; }
    sleep 1
done

run_test() {
    TOTAL=$((TOTAL+1))
    if [ "$2" = PASS ]; then PASS=$((PASS+1)); echo "  PASS: $1"
    else FAIL=$((FAIL+1)); echo "  FAIL: $1 — $3"; fi
}
status() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
launch() { curl -fsSL "$BASE/launch?$1" | HOME="$T/home" PATH="$T/bin:$PATH" sh >"$T/run.log" 2>&1; }

echo "[1] refusals"
run_test "unknown agent is a 400" "$([ "$(status "$BASE/launch?agent=nope")" = 400 ] && echo PASS)" "$(status "$BASE/launch?agent=nope")"
run_test "an agent the console does not offer is a 400" "$([ "$(status "$BASE/launch?agent=codex")" = 400 ] && echo PASS)" "$(status "$BASE/launch?agent=codex")"
run_test "an unknown model is a 404" "$([ "$(status "$BASE/launch?agent=pi&model=org%2Fnope")" = 404 ] && echo PASS)" ""
run_test "a quote-breaking Host is a 400" "$([ "$(status -H "Host: x'y" "$BASE/launch?agent=pi")" = 400 ] && echo PASS)" ""

echo "[2] the script targets the host the request reached"
S=$(curl -fsS -H 'Host: 10.0.0.5:11234' "$BASE/launch?agent=pi")
if echo "$S" | grep -q '"baseUrl": "http://10.0.0.5:11234/v1"' && echo "$S" | sh -n; then
    run_test "Host header becomes the base URL; the script parses" PASS
else
    run_test "Host header becomes the base URL; the script parses" FAIL "$(echo "$S" | head -5)"
fi

echo "[3] pi: configs written, skill linked, agent started"
launch "agent=pi&model=org%2Fchat-b"
OUT=$(cat "$T/home/agent.out" 2>/dev/null)
OK=1
echo "$OUT" | grep -q "url=$BASE " || OK=0
echo "$OUT" | grep -q 'dir=.*/.mlx-serve/pi ' || OK=0
echo "$OUT" | grep -q 'args=--provider mlx --model org/chat-b$' || OK=0
grep -q "\"baseUrl\": \"$BASE/v1\"" "$T/home/.mlx-serve/pi/models.json" 2>/dev/null || OK=0
[ -f "$T/home/.mlx-serve/pi/settings.json" ] || OK=0
[ -f "$T/home/.mlx-serve/pi/skills/mlx-serve/SKILL.md" ] || OK=0
if [ "$OK" = 1 ]; then run_test "pi launched with its models.json + skill" PASS
else run_test "pi launched with its models.json + skill" FAIL "$OUT $(cat "$T/run.log")"; fi

echo "[4] a rerun keeps an edited skill"
echo edited > "$T/home/.mlx-serve/skills/mlx-serve/SKILL.md"
launch "agent=pi"
if [ "$(cat "$T/home/.mlx-serve/skills/mlx-serve/SKILL.md")" = edited ] && grep -q 'args=--provider mlx --model org/chat-' "$T/home/agent.out"; then
    run_test "edited skill kept, default model picked" PASS
else
    run_test "edited skill kept, default model picked" FAIL "$(cat "$T/run.log")"
fi

echo "[5] opencode2: v2 binary resolved in the login shell"
launch "agent=opencode2&model=org%2Fchat-a"
OUT=$(cat "$T/home/agent.out" 2>/dev/null)
if echo "$OUT" | grep -q 'xdg=.*/.mlx-serve/opencode2 args=--standalone' \
    && grep -q "\"metricsUrl\":\"$BASE/metrics.json\"" "$T/home/.mlx-serve/opencode2/opencode/cli.json" 2>/dev/null \
    && [ -f "$T/home/.mlx-serve/opencode2/opencode/plugins/mlx-serve/tui.tsx" ]; then
    run_test "opencode 2.x started standalone with cli.json + plugin" PASS
else
    run_test "opencode 2.x started standalone with cli.json + plugin" FAIL "$OUT $(cat "$T/run.log")"
fi

echo ""
echo "$PASS/$TOTAL passed"
[ "$FAIL" -eq 0 ]
