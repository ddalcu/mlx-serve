#!/bin/bash
# Integration test: launch flags survive a COLD load (hot model switch,
# /v1/load-model, first request naming an unloaded model).
#
# Three rounds of the same bug have shipped — prefix-cache settings, then MTP +
# llama settings, then the drafter group — each time a flag reached only the
# `--model` primary while `ensureLoaded`'s LoadRequest used a struct default.
# `--no-drafter` became load-bearing on this path when `resolveInDirDrafter`
# started probing `<model_dir>/drafter` at load: a server launched with
# speculation OFF re-enabled it on every model it switched to.
#
# The drafter sidecar here is a config.json only — the probe reads nothing else,
# so the arm that probes fails its load loudly and the arm that doesn't is
# silent. That is the observable difference, and it needs no real drafter.

set -u

MODEL=${1:-~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
PORT=${2:-8131}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

MODEL=$(eval echo "$MODEL")
if [ ! -d "$MODEL" ]; then echo "SKIP: model not found at $MODEL"; exit 0; fi
if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

# Scratch root holding a symlink clone of $MODEL plus an in-dir drafter the
# probe will accept. Symlinks so the clone costs nothing (every size scan in
# the server stats THROUGH symlinks — that is its own guard).
ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mlxserve-coldflags.XXXXXX")
CLONE="$ROOT/scratch-org/cold-load-probe"
mkdir -p "$CLONE/drafter"
for f in "$MODEL"/*; do ln -s "$f" "$CLONE/$(basename "$f")"; done
cat > "$CLONE/drafter/config.json" <<'JSON'
{"model_type":"probe_assistant","block_size":4,"mask_token_id":1,"target_layer_ids":[0,1]}
JSON
# `kill 0` signals the whole process GROUP — i.e. this script. Guard on
# a non-empty pid, since cold_load_with clears it after each arm.
cleanup_all() { [ -n "${SERVER_PID:-}" ] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }; rm -rf "$ROOT"; }
trap cleanup_all EXIT

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

# Boots with $MODEL as the primary, then COLD-loads the clone by name. Prints
# the clone's load log.
cold_load_with() { # $1 = extra launch flags, $2 = log path
    ./zig-out/bin/mlx-serve --model "$MODEL" --serve --port $PORT --host 127.0.0.1 \
        --log-level info --model-dir "$ROOT" $1 >"$2" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 60); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && break
        if [ "$i" -eq 60 ]; then echo "server did not start"; return 1; fi
        sleep 1
    done
    # /v1/load-model, and the status is checked: a typo'd path 404s, and the
    # request would STILL cold-load the model named in the body, so a silent
    # `curl -sf` here leaves the arms differing for a reason this test does
    # not name.
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 900 "$BASE/v1/load-model" \
        -H "Content-Type: application/json" -d '{"model":"scratch-org/cold-load-probe"}')
    echo "  (load-model -> HTTP $code)"
    kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; SERVER_PID=
}

echo "=== launch flags survive a cold load ==="

# Baseline: no flag → the in-dir probe runs on the cold-loaded model. Its load
# fails (config-only sidecar), which is the observable proof the probe fired.
cold_load_with "" "$ROOT/plain.log" || exit 1
if grep -q "DFlash" "$ROOT/plain.log"; then
    run_test "in-dir drafter probed on a cold load (baseline)" PASS ""
else
    run_test "in-dir drafter probed on a cold load (baseline)" FAIL "no DFlash line — test setup is not exercising the probe"
fi

# --no-drafter must reach the cold path: no probe, no load attempt at all.
cold_load_with "--no-drafter" "$ROOT/nodrafter.log" || exit 1
if grep -q "DFlash" "$ROOT/nodrafter.log"; then
    run_test "--no-drafter survives a cold load" FAIL "cold-loaded model still probed: $(grep -m1 DFlash "$ROOT/nodrafter.log")"
else
    run_test "--no-drafter survives a cold load" PASS ""
fi

# Non-vacuity: "no DFlash line" is also what a cold load that never HAPPENED
# looks like. The --no-drafter arm must show the clone reaching `ready`.
if grep -q "model id=scratch-org/cold-load-probe ready" "$ROOT/nodrafter.log"; then
    run_test "the --no-drafter arm actually cold-loaded the clone" PASS ""
else
    run_test "the --no-drafter arm actually cold-loaded the clone" FAIL "no ready line — the silent arm proves nothing"
fi

echo ""
echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
