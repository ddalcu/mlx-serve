#!/bin/bash
# website/agent.sh, the copy-paste installer on mlxserve.com (`curl -fsSL https://mlxserve.com/agent.sh | sh`):
# asks for the mlx-serve address and the coding agent (installed ones marked), runs the agent's own
# installer when it is missing, then the server's `/launch?agent=<id>` script, which configures the
# agent with the mlx-serve skill and starts it.
#
# FULLY HERMETIC: stub models, fake agents, installers and login shell on PATH (a fake curl serves the
# installers and passes every other URL to the real one), scratch HOME.
#
# Usage: ./tests/test_website_agent_install.sh [port]

set -u
source "$(dirname "$0")/lib/server.sh"

PORT=${1:-8134}
EMPTY_PORT=$((PORT + 1))
BASE="http://127.0.0.1:$PORT"
BIN=./zig-out/bin/mlx-serve
SCRIPT=website/agent.sh
PASS=0
FAIL=0
TOTAL=0

[ -x "$BIN" ] || { echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"; exit 1; }

T=$(mktemp -d)
mkdir -p "$T/root/org/chat-a" "$T/empty" "$T/server-home" "$T/base" "$T/agents" "$T/opencode"
printf '{"model_type":"qwen3"}' > "$T/root/org/chat-a/config.json"

# $T/base: what every run gets — a login shell that keeps PATH, and a curl that serves the agents'
# installers (each one records itself and drops its agent into a user bin dir that is NOT on PATH).
cat > "$T/base/zsh" <<'EOF'
#!/bin/sh
[ "$1" = -l ] && shift
[ "$1" = -c ] && shift
exec /bin/sh -c "$1"
EOF
cat > "$T/agent" <<'EOF'
#!/bin/sh
echo "$(basename "$0") dir=${PI_CODING_AGENT_DIR:-} url=${MLX_SERVE_URL:-} anthropic=${ANTHROPIC_BASE_URL:-} tty=$(tty) args=$*" > "$HOME/agent.out"
EOF
cat > "$T/base/curl" <<EOF
#!/bin/sh
out=
for a; do [ "\$prev" = -o ] && out=\$a; prev=\$a; done
for a; do
    case \$a in
        https://pi.dev/install.sh) bin=.pi/agent/bin cmd=pi ;;
        https://claude.ai/install.sh) bin=.local/bin cmd=claude ;;
        https://omp.sh/install) bin=.local/bin cmd=omp ;;
        *) continue ;;
    esac
    printf 'echo "%s" >> "\$HOME/installers.out"; mkdir -p "\$HOME/%s"; cp "$T/agent" "\$HOME/%s/%s"\n' "\$a" "\$bin" "\$bin" "\$cmd" > "\${out:-/dev/stdout}"
    exit 0
done
exec /usr/bin/curl "\$@"
EOF
cp "$T/agent" "$T/agents/pi"
cp "$T/agent" "$T/agents/claude"
cat > "$T/opencode/opencode" <<'EOF'
#!/bin/sh
[ "$1" = --version ] && { echo "2.1.0"; exit 0; }
echo "opencode xdg=$XDG_CONFIG_HOME args=$*" > "$HOME/agent.out"
EOF
chmod +x "$T/agent" "$T"/base/* "$T"/agents/* "$T"/opencode/*

HOME="$T/server-home" "$BIN" serve --host 127.0.0.1 --port "$PORT" --model-dir "$T/root" --log-file off >"$T/server.log" 2>&1 &
SERVER_PID=$!
HOME="$T/server-home" "$BIN" serve --host 127.0.0.1 --port "$EMPTY_PORT" --model-dir "$T/empty" --log-file off >"$T/empty.log" 2>&1 &
EMPTY_PID=$!
cleanup() { kill $SERVER_PID $EMPTY_PID 2>/dev/null; wait $SERVER_PID $EMPTY_PID 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
for p in "$PORT" "$EMPTY_PORT"; do
    srv_wait_health "http://127.0.0.1:$p" 30 1 || { echo "FAIL: server on $p did not start"; cat "$T/server.log" "$T/empty.log"; exit 1; }
done

run_test() {
    TOTAL=$((TOTAL+1))
    if [ "$2" = PASS ]; then PASS=$((PASS+1)); echo "  PASS: $1"
    else FAIL=$((FAIL+1)); echo "  FAIL: $1 — $3"; fi
}
fresh() { rm -rf "$T/home"; mkdir -p "$T/home"; }
# install <extra PATH dirs> [args...]: the installer as `curl … | sh -s -- args` runs it, stdin a pipe
install() { local dirs=$1; shift; HOME="$T/home" PATH="$T/base:$dirs:/usr/bin:/bin" sh -s -- "$@" < "$SCRIPT" >"$T/run.log" 2>&1; }

echo "[1] a bare host gets mlx-serve's port, and an unreachable one is named"
fresh
install "$T/agents" mlx.invalid pi; CODE=$?
if [ "$CODE" != 0 ] && grep -q 'http://mlx.invalid:11234' "$T/run.log" && [ ! -f "$T/home/agent.out" ]; then
    run_test "exits non-zero naming http://mlx.invalid:11234, starts nothing" PASS
else
    run_test "exits non-zero naming http://mlx.invalid:11234, starts nothing" FAIL "exit $CODE: $(cat "$T/run.log")"
fi

echo "[2] an installed agent: configured for the server, skill linked, started"
install "$T/agents" "127.0.0.1:$PORT" pi; CODE=$?
OUT=$(cat "$T/home/agent.out" 2>/dev/null)
if [ "$CODE" = 0 ] && echo "$OUT" | grep -q "^pi .*url=$BASE " && echo "$OUT" | grep -q 'dir=.*/.mlx-serve/pi ' \
    && grep -q "\"baseUrl\": \"$BASE/v1\"" "$T/home/.mlx-serve/pi/models.json" 2>/dev/null \
    && [ -f "$T/home/.mlx-serve/pi/skills/mlx-serve/SKILL.md" ] && [ ! -f "$T/home/installers.out" ]; then
    run_test "pi started on $BASE with its models.json and the mlx-serve skill, nothing installed" PASS
else
    run_test "pi started on $BASE with its models.json and the mlx-serve skill, nothing installed" FAIL "exit $CODE: $OUT $(cat "$T/run.log")"
fi

echo "[3] on a terminal: answers typed at the prompts; Enter keeps the last server and agent"
# expect types at each prompt like a person (script(1) with piped input sends EOF to the first read)
# Agents built on Bun (Claude Code, oh-my-pi, OpenCode) and Grok cannot watch a reopened /dev/tty on macOS.
real_tty() { grep -q 'tty=/dev/' "$T/home/agent.out" && ! grep -q 'tty=/dev/tty ' "$T/home/agent.out"; }
tty_run() {
    HOME="$T/home" PATH="$T/base:$T/agents:/usr/bin:/bin" expect -c "
        set timeout 30
        log_user 1
        spawn sh $SCRIPT
        expect {mlx-serve address \\[*\\]: } { send -- \"$1\r\" }
        expect {Agent \\[*\\]: } { send -- \"$2\r\" }
        expect eof
    " >"$T/run.log" 2>&1
}
if command -v expect >/dev/null 2>&1; then
    fresh
    tty_run "127.0.0.1:$PORT" 2
    if grep -q '1) pi .*installed' "$T/run.log" && grep -q '2) Claude Code .*installed' "$T/run.log" \
        && grep -q "^claude .*anthropic=$BASE" "$T/home/agent.out" 2>/dev/null && real_tty; then
        run_test "the address and agent 2 typed in: Claude Code starts on $BASE, on the terminal itself" PASS
    else
        run_test "the address and agent 2 typed in: Claude Code starts on $BASE, on the terminal itself" FAIL "$(cat "$T/run.log" "$T/home/agent.out" 2>/dev/null)"
    fi
    rm -f "$T/home/agent.out"
    tty_run "" ""
    if grep -q "address \[$BASE\]" "$T/run.log" && grep -q 'Agent \[2\]' "$T/run.log" \
        && grep -q "^claude .*anthropic=$BASE" "$T/home/agent.out" 2>/dev/null; then
        run_test "Enter twice: the same server and agent again" PASS
    else
        run_test "Enter twice: the same server and agent again" FAIL "$(cat "$T/run.log")"
    fi
    rm -f "$T/home/agent.out"
    # piped in like `curl … | sh`: the script arrives on stdin, the terminal is only stderr
    HOME="$T/home" PATH="$T/base:$T/agents:/usr/bin:/bin" expect -c "
        set timeout 30
        spawn sh -c {sh -s -- 127.0.0.1:$PORT pi < $SCRIPT}
        expect eof
    " >"$T/run.log" 2>&1
    if grep -q "^pi .*url=$BASE " "$T/home/agent.out" 2>/dev/null && real_tty; then
        run_test "piped in, the agent still gets the terminal itself as stdin" PASS
    else
        run_test "piped in, the agent still gets the terminal itself as stdin" FAIL "$(cat "$T/run.log" "$T/home/agent.out" 2>/dev/null)"
    fi
else
    echo "  SKIP: no expect"
fi

echo "[4] a missing agent: its own installer runs, its command is found off PATH"
fresh
HOME="$T/home" PATH="$T/base:/usr/bin:/bin" MLX_SERVE_URL="$BASE" MLX_SERVE_AGENT=pi sh < "$SCRIPT" >"$T/run.log" 2>&1; CODE=$?
if [ "$CODE" = 0 ] && [ "$(cat "$T/home/installers.out" 2>/dev/null)" = https://pi.dev/install.sh ] \
    && grep -q "^pi .*url=$BASE " "$T/home/agent.out" 2>/dev/null; then
    run_test "pi's installer ran, pi started from ~/.pi/agent/bin" PASS
else
    run_test "pi's installer ran, pi started from ~/.pi/agent/bin" FAIL "exit $CODE: $(cat "$T/run.log")"
fi

echo "[5] with no agent named, the default is the first one already installed"
fresh
install "$T/agents" "127.0.0.1:$PORT"; CODE=$?
if [ "$CODE" = 0 ] && grep -q "^pi " "$T/home/agent.out" 2>/dev/null && [ ! -f "$T/home/installers.out" ]; then
    run_test "pi (installed) is picked over installing anything" PASS
else
    run_test "pi (installed) is picked over installing anything" FAIL "exit $CODE: $(cat "$T/run.log")"
fi
fresh
rm "$T/agents/pi"
install "$T/agents" "127.0.0.1:$PORT"; CODE=$?
if [ "$CODE" = 0 ] && grep -q "^claude .*anthropic=$BASE" "$T/home/agent.out" 2>/dev/null && [ ! -f "$T/home/installers.out" ]; then
    run_test "Claude Code (installed) is picked when pi is not" PASS
else
    run_test "Claude Code (installed) is picked when pi is not" FAIL "exit $CODE: $(cat "$T/run.log")"
fi
cp "$T/agent" "$T/agents/pi"

echo "[6] an agent with no installer here is offered only once installed"
fresh
install "$T/agents" "127.0.0.1:$PORT" opencode2; CODE=$?
if [ "$CODE" != 0 ] && grep -q 'opencode2' "$T/run.log" && grep -q 'pi' "$T/run.log" && [ ! -f "$T/home/agent.out" ]; then
    run_test "absent OpenCode 2 is refused, naming the agents on offer" PASS
else
    run_test "absent OpenCode 2 is refused, naming the agents on offer" FAIL "exit $CODE: $(cat "$T/run.log")"
fi
install "$T/agents:$T/opencode" "127.0.0.1:$PORT" opencode2; CODE=$?
if [ "$CODE" = 0 ] && grep -q '^opencode xdg=.*/.mlx-serve/opencode2 ' "$T/home/agent.out" 2>/dev/null; then
    run_test "an opencode answering 2.x is OpenCode 2, and starts" PASS
else
    run_test "an opencode answering 2.x is OpenCode 2, and starts" FAIL "exit $CODE: $(cat "$T/run.log")"
fi

echo "[7] the server's refusal reaches the user"
fresh
install "$T/agents" "127.0.0.1:$EMPTY_PORT" pi; CODE=$?
if [ "$CODE" != 0 ] && grep -q 'no such chat model' "$T/run.log" && [ ! -f "$T/home/agent.out" ]; then
    run_test "a server with no chat model says so" PASS
else
    run_test "a server with no chat model says so" FAIL "exit $CODE: $(cat "$T/run.log")"
fi

echo ""
echo "$PASS/$TOTAL passed"
[ "$FAIL" -eq 0 ]
