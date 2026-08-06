#!/usr/bin/env bash
# Registry eviction gate vs. a staged-residency media model (#126).
#
# The gate reserves a model's post-load bytes BEFORE the media preflight runs,
# and it used to bill every entry the sum of every safetensors in its dir. For
# MiniMax-H3 that sum (37.55 GiB) is 14.7 GiB more than the model can ever
# hold — the text encoder runs and is FREED before the DiT loads, so the real
# staged peak is 22.83 GiB — and on a 48 GB Mac (auto cap 30.0 GiB) it refused
# every load, permanently, on an idle server with nothing to evict.
#
# Hermetic: the "model" is four SPARSE files with the real pack's byte sizes and
# a config.json naming the backend. Nothing is ever read, so the load fails at
# engine build — which is exactly the point. WHICH failure it is, is the test:
# past the gate the answer is no longer the gate's 503.
#
# Usage: ./tests/test_media_eviction_gate.sh [port]
set -uo pipefail
PORT="${1:-11374}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/mlxserve-gate.XXXXXX")"
SRV=""
# `kill ""` is a no-op; `kill 0` signals the whole PROCESS GROUP, i.e. this
# shell — which is how the cleanup path exited 144 with every check passing.
trap 'rm -rf "$TMP"; [ -n "$SRV" ] && kill "$SRV" 2>/dev/null' EXIT
DIR="$TMP/models/fake/MiniMax-H3-Gate"
mkdir -p "$DIR"

# The real ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit byte sizes, sparse.
sparse() { dd if=/dev/zero of="$DIR/$1" bs=1 count=0 seek="$2" 2>/dev/null; }
sparse text_encoder.safetensors 15804791921
sparse transformer.safetensors  18698813290
sparse video_vae.safetensors     5207808496
sparse audio_vae.safetensors      605254808
cat >"$DIR/config.json" <<'JSON'
{"model_type":"minimax_h3","partition":"fl2va","tasks":["t2va","fl2va"],"fps":24}
JSON
# sum = 40,316,668,515 (37.55 GiB); staged peak = 24,511,876,594 (22.83 GiB)

MODEL_ID="fake/MiniMax-H3-Gate"
rc=0

boot() { # boot <cap> <logfile>
  "$BIN" --serve --port "$PORT" --model-dir "$TMP/models" --max-resident-mem "$1" >"$2" 2>&1 &
  SRV=$!
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -8 "$2"; return 1; }
    sleep 0.5
  done
  echo "FAIL: server never became healthy"; return 1
}
stop() { [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }; SRV=""; }

# ── [1] A cap ABOVE the staged peak but BELOW the dir sum must not refuse.
# 30 GB is exactly what a 48 GB Mac's `auto` resolves to, i.e. the reported bug.
LOG1="$TMP/above.log"
boot 30GB "$LOG1" || exit 1
BODY=$(curl -s -o "$TMP/b1.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/load-model" \
  -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL_ID\"}")
if [ "$BODY" = "503" ] && grep -q out_of_memory "$TMP/b1.json"; then
  echo "FAIL: gate refused a load that fits (503 out_of_memory at a 30 GB cap)"; rc=1
else
  echo "PASS: staged-residency model clears the gate at a 30 GB cap (got $BODY)"
fi
# It must also not have LOGGED a refusal — a silent pass here would mean the
# gate refused and something else answered.
if grep -q "Refusing to load" "$LOG1"; then
  echo "FAIL: gate logged a refusal at a cap above the staged peak"; rc=1
fi
stop

# ── [2] A cap BELOW the staged peak still refuses — and says what and how.
LOG2="$TMP/below.log"
boot 8GB "$LOG2" || exit 1
CODE=$(curl -s -o "$TMP/b2.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/load-model" \
  -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL_ID\"}")
if [ "$CODE" = "503" ] && grep -q out_of_memory "$TMP/b2.json"; then
  echo "PASS: a genuinely-too-small cap still refuses (503)"
else
  echo "FAIL: expected 503 out_of_memory under an 8 GB cap, got $CODE"; rc=1
fi
# The refusal must name the knob, in the body and in the log. "retry after
# current requests complete" pointed at concurrency on an idle server.
grep -q -- "--max-resident-mem" "$TMP/b2.json" \
  && echo "PASS: the 503 body names --max-resident-mem" \
  || { echo "FAIL: the 503 body does not name the flag"; rc=1; }
grep -q "retry after current requests" "$TMP/b2.json" \
  && { echo "FAIL: the 503 still blames concurrency"; rc=1; } \
  || echo "PASS: the 503 no longer blames concurrency"
if grep -q "Refusing to load .*needs ~.*--max-resident-mem" "$LOG2"; then
  echo "PASS: the refusal is logged with its numbers"
else
  echo "FAIL: refusal logged nothing actionable"; grep -i "memory\|refus" "$LOG2" | tail -3; rc=1
fi
# The number it names must be the STAGED peak (~22.8 GB + 10%), not the sum.
EST=$(sed -E -n 's/.*needs ~([0-9.]+) GB.*/\1/p' "$LOG2" | head -1)
if [ -n "$EST" ] && python3 -c "import sys; sys.exit(0 if 24.0 < float('$EST') < 29.0 else 1)"; then
  echo "PASS: the gate billed the staged peak (~$EST GB, not the ~41.3 sum)"
else
  echo "FAIL: gate estimate '$EST' GB is not staged peak + 10%"; rc=1
fi
stop

[ $rc -eq 0 ] && echo "ALL PASS" || echo "SOME FAILURES"
exit $rc
