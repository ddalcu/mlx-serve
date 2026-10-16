#!/bin/bash
# A dense DFlash slot keeps drafting beside a few other streams (env-gated on a dense target +
# DFlash2 sidecar pair, Qwen3.8-27B today):
#
#   DFLASH_TEST_MODEL=<27B dir> DFLASH_TEST_DRAFTER=<DFlash2 dir> ./tests/test_dflash_company.sh
#
# Three simultaneous greedy requests: the slot admitted first arms DFlash, the others arm MTP. It must
# not yield (`dflash=disabled (company`) until the others are a crowd. Engagement counts from the log,
# never output shape. `MLX_SERVE_BINARY` points the test at another build, e.g. main for the red run.

set -euo pipefail

MODEL="${DFLASH_TEST_MODEL:-}"
DRAFTER="${DFLASH_TEST_DRAFTER:-}"
if [ -z "$MODEL" ] || [ -z "$DRAFTER" ]; then
    echo "SKIP: DFLASH_TEST_MODEL / DFLASH_TEST_DRAFTER not set"
    exit 0
fi
for d in "$MODEL" "$DRAFTER"; do
    [ -f "$d/config.json" ] || { echo "FAIL: $d/config.json not found"; exit 1; }
done

PORT="${DFLASH_TEST_PORT:-11354}"
BASE="http://127.0.0.1:$PORT"
BIN="${MLX_SERVE_BINARY:-$(dirname "$0")/../zig-out/bin/mlx-serve}"
LOG=$(mktemp /tmp/dflash_company_serve.XXXXXX)
OUT=$(mktemp -d /tmp/dflash_company_out.XXXXXX)

"$BIN" --model "$MODEL" --drafter "$DRAFTER" --serve --host 127.0.0.1 --port "$PORT" \
    --max-concurrent 8 --prefill-decode-share 0.5 > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$OUT"; }
trap cleanup EXIT

MODEL_MB=$(du -sm "$MODEL" 2>/dev/null | awk '{print $1}')
READY_SECS=$(( 300 + ${MODEL_MB:-0} / 100 ))
echo "waiting for model load (up to ${READY_SECS}s)..."
for _ in $(seq 1 $((READY_SECS / 3))); do
    grep -q "Model ready (loaded on inference thread)" "$LOG" && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "FAIL: server died during load"; tail -20 "$LOG"; exit 1; }
    sleep 3
done
grep -q "Model ready (loaded on inference thread)" "$LOG" || { echo "FAIL: model did not load"; tail -20 "$LOG"; exit 1; }

pass=0; fail=0
ok()  { echo "PASS $1"; pass=$((pass+1)); }
bad() { echo "FAIL $1"; shift; for line in "$@"; do echo "  $line"; done; fail=$((fail+1)); }

req() { # index prompt: completion token count to $OUT/<index>
    curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
        \"model\": \"mlx-serve\", \"temperature\": 0.0, \"max_tokens\": 300,
        \"messages\": [{\"role\": \"user\", \"content\": \"$2\"}]
    }" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["completion_tokens"])' > "$OUT/$1"
}

# A request after load is cold: send it alone so the burst below starts from a warm server.
req 0 "Say hello." || true

# Wait on the three clients by pid: a bare `wait` also waits on the server.
req 1 "Write a short essay about the history of quantum computing." & p1=$!
req 2 "Explain how a compiler turns source code into machine code." & p2=$!
req 3 "Describe the water cycle for a graduate seminar." & p3=$!
wait "$p1" "$p2" "$p3"

for i in 1 2 3; do
    n=$(cat "$OUT/$i" 2>/dev/null || echo 0)
    if [ "${n:-0}" -ge 100 ]; then ok "stream $i generated $n tokens"; else bad "stream $i generated $n tokens"; fi
done

# [1] DFlash armed and ran rounds, so a zero below is not a build where it never engaged.
if grep -q "spec-stats\] mode=dflash attempts=[1-9]" "$LOG"; then ok "dflash rounds engaged"; else bad "dflash rounds engaged" "$(grep spec-stats "$LOG" | head -3)"; fi

# [2] Three streams are fewer than a crowd: the DFlash slot never yields to company.
YIELDS=$(grep -c "dflash=disabled (company" "$LOG" || true)
if [ "$YIELDS" -eq 0 ]; then ok "no yield to company at 3 streams"; else bad "no yield to company at 3 streams" "$YIELDS yield lines" "$(grep 'dflash=disabled (company' "$LOG" | head -2)"; fi

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
