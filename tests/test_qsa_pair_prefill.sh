#!/usr/bin/env bash
# Disposable mixed-pack integration server; terminates only its own child.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}
PORT=${QSA_PAIR_TEST_PORT:-18765}
MODEL=${QWEN4_MODEL:-ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit}
PAIR=${QSA_PAIR_TEST_PAIR:-on}
OUT=${QSA_PAIR_TEST_OUT:-$(mktemp -d)}
mkdir -p "$OUT"
LOG="$OUT/server.log"
python3 - "$PORT" <<'CHECK'
import socket,sys
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',int(sys.argv[1]))) == 0:
        raise SystemExit('Refusing to replace a listening server')
CHECK
case "$PAIR" in on) VALUE=1;; off) VALUE=0;; *) echo 'QSA_PAIR_TEST_PAIR must be on or off' >&2; exit 2;; esac
owner="qsa-pair-test-$$"
server_pid=''
cleanup() {
    if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
    if [[ -n "${MLX_SERVE_TEST_GPU_LOCK:-}" ]]; then "$MLX_SERVE_TEST_GPU_LOCK" release "$owner"; fi
}
if [[ -n "${MLX_SERVE_TEST_GPU_LOCK:-}" ]]; then "$MLX_SERVE_TEST_GPU_LOCK" acquire "$owner"; fi
trap cleanup EXIT
DYLD_LIBRARY_PATH="$ROOT/lib/mlx/lib:$ROOT/lib/llama/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
MLX_SERVE_PREFILL_CHUNK=8192 MLX_SERVE_QSA_PAIR="$VALUE" \
MLX_SERVE_HC_PREFILL=1 MLX_SERVE_GDN_PREFILL_FUSED=1 \
"$BIN" serve --host 127.0.0.1 --port "$PORT" --ctx-size 131072 \
    --prefill-chunk 8192 --prefix-cache-entries 0 --kv-quant off --no-mtp --no-pld --no-drafter >"$LOG" 2>&1 &
server_pid=$!
for ((i=0; i<150; i++)); do
    kill -0 "$server_pid" 2>/dev/null || { cat "$LOG"; exit 1; }
    if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
    sleep 0.2
done
extra=()
[[ "$PAIR" == on ]] && extra+=(--first-chunk)
python3 "$ROOT/tests/test_qsa_pair_prefill.py" --url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --log "$LOG" --pair "$PAIR" --hc on --gdn on "${extra[@]}"
printf 'Logs: %s\n' "$LOG"
