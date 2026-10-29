#!/bin/bash
# test_load_failure_no_fallback.sh — a pack that failed to load is never answered by another model. The model is
# cloned (APFS, instant) with one tensor renamed in the safetensors header (same length, offsets hold): a failed
# `--model` load exits non-zero in the app's launch shape; on a server booted with the GOOD copy the broken pack named
# by id, path or path-with-slash on chat and /v1/messages answers the named 500, an unregistered path is a 404, and
# the good path, the `mlx-serve` alias and an SDK name still answer.
#
#   ./tests/test_load_failure_no_fallback.sh [model_dir] [port]
#
# Hermetic twin: resolveRequestModelId.

set -u
source "$(dirname "$0")/lib/server.sh"

MODEL=${1:-~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}
PORT=${2:-8133}
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
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq is required"; exit 1; }

WORK=$(mktemp -d /tmp/mlx-serve-585.XXXXXX)
GOOD="$WORK/test/good"
BROKEN="$WORK/test/broken"
mkdir -p "$GOOD" "$BROKEN"
for f in "$MODEL"/*; do
    real=$(readlink -f "$f")
    ln -s "$real" "$GOOD/$(basename "$f")"
    case "$f" in *.safetensors) cp -c "$real" "$BROKEN/$(basename "$f")" 2>/dev/null || cp "$real" "$BROKEN/$(basename "$f")" ;;
                 *) ln -s "$real" "$BROKEN/$(basename "$f")" ;; esac
done
python3 - "$BROKEN" <<'EOF' || { echo "FAIL: could not break a tensor name"; rm -rf "$WORK"; exit 1; }
import glob, json, struct, sys
for p in sorted(glob.glob(sys.argv[1] + "/*.safetensors")):
    with open(p, "r+b") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = f.read(n)
        names = [k for k in json.loads(h) if ".layers.1." in k and k.endswith(".weight")]
        if not names:
            continue
        old = ('"%s"' % names[0]).encode()
        h = h.replace(old, old[:-2] + b'X"', 1)
        f.seek(8); f.write(h)
        print("broke", names[0])
        sys.exit(0)
sys.exit(1)
EOF

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

echo "=== a failed load is never answered by another model ==="

# The app's launch shape: the broken pack as --model beside a good one in --model-dir.
./zig-out/bin/mlx-serve --model "$BROKEN" --model-dir "$WORK" --serve --port $PORT --host 127.0.0.1 \
    >/tmp/mlx-serve-585-boot.log 2>&1 &
BOOT_PID=$!
for i in $(seq 1 300); do kill -0 $BOOT_PID 2>/dev/null || break; sleep 1; done
if kill -0 $BOOT_PID 2>/dev/null; then
    kill $BOOT_PID; wait $BOOT_PID 2>/dev/null
    run_test "a failed --model load exits the process" FAIL "still running after 300s"
else
    wait $BOOT_PID; RC=$?
    if [ "$RC" -ne 0 ] && grep -q "MissingWeight" /tmp/mlx-serve-585-boot.log; then
        run_test "a failed --model load exits the process" PASS ""
    else
        run_test "a failed --model load exits the process" FAIL "exit $RC: $(tail -2 /tmp/mlx-serve-585-boot.log)"
    fi
fi

./zig-out/bin/mlx-serve --model "$GOOD" --model-dir "$WORK" --serve --port $PORT --host 127.0.0.1 \
    --log-level info >/tmp/mlx-serve-585.log 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT

srv_wait_health "$BASE" 300 1 || { echo "FAIL: server did not start within 300s"; exit 1; }


# expect <label> <url path> <json body> <want http code> [want body substring]
expect() {
    CODE=$(curl -s -o /tmp/mlx-serve-585-body.json -w '%{http_code}' -m 600 "$BASE$2" \
        -H "Content-Type: application/json" -d "$3")
    BODY=$(head -c 200 /tmp/mlx-serve-585-body.json)
    if [ "$CODE" = "$4" ] && { [ -z "${5:-}" ] || grep -q "$5" /tmp/mlx-serve-585-body.json; }; then
        run_test "$1" PASS ""
    else
        run_test "$1" FAIL "got HTTP $CODE: $BODY"
    fi
}

chat() { echo "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2}"; }

expect "load-model on the broken pack names the failure" /v1/load-model \
    "{\"model\":\"$BROKEN\",\"default\":true}" 500 MissingWeight
expect "chat by the broken pack's PATH names the failure" /v1/chat/completions "$(chat "$BROKEN")" 500 MissingWeight
expect "chat by the broken pack's PATH/ names the failure" /v1/chat/completions "$(chat "$BROKEN/")" 500 MissingWeight
expect "chat by the broken pack's id names the failure" /v1/chat/completions "$(chat test/broken)" 500 MissingWeight
expect "anthropic messages by PATH names the failure" /v1/messages \
    "{\"model\":\"$BROKEN\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" 500 MissingWeight
expect "chat by an unregistered PATH is a 404" /v1/chat/completions "$(chat "$WORK/test/nothing-here")" 404 model_not_found

# The rest still works, and the failed load did not steal the default.
expect "chat by the good pack's PATH answers" /v1/chat/completions "$(chat "$GOOD")" 200
expect "the mlx-serve alias still answers" /v1/chat/completions "$(chat mlx-serve)" 200
expect "an SDK name still falls back to the default" /v1/chat/completions "$(chat gpt-4)" 200

echo ""
echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
