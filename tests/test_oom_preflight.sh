#!/bin/bash
# Regression test for the GPU-OOM pre-flight refusal path (issue #47).
#
# When the #45 memory pre-flight refuses a load (error.InsufficientMemory),
# main() returns the error and unwinds its defers. A redundant errdefer+defer
# pair (identical bodies) for config/tok/chat_config used to free each twice on
# that error path → double-free → SIGSEGV during shutdown, AFTER the legible
# error was printed. This pins the fix: the refusal must exit cleanly (no
# signal-11), not crash.
#
# The pre-flight sums *.safetensors sizes via statFile (it never opens them), so
# a SPARSE dummy .safetensors inflates the apparent weight size past free RAM
# and forces a deterministic refusal — no real memory pressure required.
#
# Usage: ./tests/test_oom_preflight.sh [model_dir] [port]
#   Needs any loadable model dir for CPU-side setup (config.json + tokenizer).

set -u

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e2b-it-8bit}"
PORT="${2:-11293}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
FAKE="/tmp/test_oom_preflight_model"
LOG=/tmp/test_oom_preflight.log
PASS=0
FAIL=0

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"; fi
}

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL"
    exit 0
fi
if [ ! -x "$BINARY" ]; then
    echo "SKIP: binary not found: $BINARY (build with: zig build -Doptimize=ReleaseFast)"
    exit 0
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

# Build the fake-OOM model: symlink the real (small) files so CPU-side setup
# succeeds, then add a huge SPARSE dummy weight file to trip the pre-flight.
rm -rf "$FAKE"; mkdir -p "$FAKE"
for f in "$MODEL"/*; do ln -sf "$f" "$FAKE/$(basename "$f")"; done
truncate -s 80G "$FAKE/zzz_oom_dummy.safetensors" 2>/dev/null || \
    mkfile -n 80g "$FAKE/zzz_oom_dummy.safetensors"

echo ""
echo "── #47: pre-flight refusal must exit cleanly (no SIGSEGV) ──"

# Run WITHOUT the bypass so the pre-flight fires. The model never loads (refused
# on the apparent 80 GB weights), so this returns within a couple seconds.
"$BINARY" --model "$FAKE" --serve --port "$PORT" --no-pld --log-level info > "$LOG" 2>&1
CODE=$?

# 139 = 128 + SIGSEGV(11) — the bug. Any non-139 (clean error exit, typically 1)
# is the fix.
check "pre-flight refusal does NOT segfault (exit != 139), got $CODE" \
    "$([ "$CODE" != "139" ] && echo 1 || echo 0)"
check "process exited non-zero (load correctly refused)" \
    "$([ "$CODE" != "0" ] && echo 1 || echo 0)"
check "legible pre-flight error was printed before exit" \
    "$(grep -qi "Insufficient memory to load model" "$LOG" && echo 1 || echo 0)"
check "no crash signature in output" \
    "$(grep -qiE "Segmentation fault|panic|EXC_BAD_ACCESS" "$LOG" && echo 0 || echo 1)"

rm -rf "$FAKE"

echo ""
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}PASS${NC} $TOTAL/$TOTAL tests passed"
    exit 0
else
    echo -e "${RED}FAIL${NC} $FAIL/$TOTAL tests failed"
    echo "--- server log (last 15 lines) ---"; tail -15 "$LOG" 2>/dev/null || true
    exit 1
fi
