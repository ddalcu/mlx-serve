#!/bin/bash
# `mlx-serve run` (TTY REPL mode) must not print `[discovery] skip …` lines.
#
# Regression: the REPL log-quieting (info → warn) sat AFTER the models-root
# discovery scan in main(), so every unsupported dir under ~/.mlx-serve/models
# (LTX, a ViT classifier, partial downloads, …) printed an info-level
# skip line into the chat REPL greeting. The quieting must take effect BEFORE
# discovery runs; an explicit --log-level keeps the diagnostics reachable.
#
# Hermetic: fake $HOME with one unsupported model dir; no weights, no load —
# `run` exits at config parse of a nonexistent model dir, after discovery has
# already scanned (and logged, or not) the models root. Needs a pty (`script`)
# because REPL mode only engages when stdin is a TTY.
set -u

BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
PORT="${1:-11321}"

if [ ! -x "$BIN" ]; then
    echo "SKIP: $BIN not found — build first: zig build -Doptimize=ReleaseFast"
    exit 0
fi

SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
FAKE_HOME="$SCRATCH/home"
mkdir -p "$FAKE_HOME/.mlx-serve/models/fake-vit-classifier"
printf '{"model_type": "vit"}\n' > "$FAKE_HOME/.mlx-serve/models/fake-vit-classifier/config.json"

# Run `mlx-serve run` under a pty, transcript to $1. Bounded wait so a
# regression can't hang the suite; the process normally exits on its own
# (config parse failure on the nonexistent model dir).
run_case() {
    local out="$1"; shift
    : > "$out"
    HOME="$FAKE_HOME" script -q "$out" "$BIN" run /nonexistent-model-dir --port "$PORT" "$@" </dev/null >/dev/null 2>&1 &
    local pid=$!
    for _ in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    return 0
}

PASS=0
FAIL=0
check() { # $1 = description, $2 = 0/1 (0 = ok)
    if [ "$2" -eq 0 ]; then
        echo "PASS: $1"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $1"
        FAIL=$((FAIL + 1))
    fi
}

# 1. Default `run`: no skip lines in the REPL transcript.
run_case "$SCRATCH/default.txt"
if grep -q "\[discovery\] skip" "$SCRATCH/default.txt"; then
    check "run (default) does not print [discovery] skip lines" 1
else
    check "run (default) does not print [discovery] skip lines" 0
fi

# 2. Explicit --log-level info: skip lines still reachable (also proves the
#    fixture actually produces them — guards against a vacuous check 1).
run_case "$SCRATCH/info.txt" --log-level info
if grep -q "\[discovery\] skip" "$SCRATCH/info.txt"; then
    check "run --log-level info still prints [discovery] skip lines" 0
else
    check "run --log-level info still prints [discovery] skip lines" 1
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
