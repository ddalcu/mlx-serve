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
# The number it names must be the BIGGEST STAGE, not the sum and not a stage
# plus another stage. For this pack: TE 14.72, DiT 17.41 x 0.65 (precomputeAdaln
# frees the 13B modulation weights) = 11.32, VAEs 5.41 — and the two stages that
# GENERATE carry a 6 GiB transient allowance the TE stage does not. So
# max(14.72, 11.32 + 6, 5.41 + 6) = 17.32, against a 37.55 dir sum.
EST=$(sed -E -n 's/.*needs ~([0-9.]+) GB.*/\1/p' "$LOG2" | head -1)
if [ -n "$EST" ] && python3 -c "import sys; sys.exit(0 if 16.0 < float('$EST') < 19.0 else 1)"; then
  echo "PASS: the gate billed the biggest stage (~$EST GB, not the ~37.6 sum)"
else
  echo "FAIL: gate estimate '$EST' GB is not the staged peak"; rc=1
fi
stop

# ── [3] Turbo LoRA request surface: both 400s fire BEFORE any weight is read,
# so the sparse pack exercises them hermetically. A missing turbo_lora file is
# a NAMED 400 (never a silent slow render — the silent-flag-eater class), and
# the 4-step floor is the distillation's own.
LOG3="$TMP/turbo.log"
boot 30GB "$LOG3" || exit 1
CODE=$(curl -s -o "$TMP/b3.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/video/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"x\",\"turbo\":true}")
if [ "$CODE" = "400" ] && grep -q "turbo_lora.safetensors" "$TMP/b3.json"; then
  echo "PASS: turbo without the LoRA file is a named 400"
else
  echo "FAIL: expected a named 400 for turbo without turbo_lora.safetensors, got $CODE: $(cat "$TMP/b3.json" | head -c 200)"; rc=1
fi
sparse turbo_lora.safetensors 744000000
CODE=$(curl -s -o "$TMP/b4.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/video/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"x\",\"turbo\":true,\"steps\":2}")
if [ "$CODE" = "400" ] && grep -q "at least 4 steps" "$TMP/b4.json"; then
  echo "PASS: turbo below the 4-step floor is a named 400"
else
  echo "FAIL: expected the 4-step-floor 400, got $CODE: $(cat "$TMP/b4.json" | head -c 200)"; rc=1
fi
# With the file present and steps legal the probe must PASS — the request then
# dies inside generate() on the sparse weights (500), which is the proof the
# 400s above were the probe and not something later.
CODE=$(curl -s -o "$TMP/b5.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/video/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"x\",\"turbo\":true}")
if [ "$CODE" != "400" ]; then
  echo "PASS: turbo with the file present clears the probe (got $CODE from the sparse weights)"
else
  echo "FAIL: turbo still 400s with turbo_lora.safetensors present: $(cat "$TMP/b5.json" | head -c 200)"; rc=1
fi

# ── [3b] Stacked style LoRAs on H3: the request surface is the SAME
# `lora_paths`/`lora_scales` grammar the image and LTX handlers take, so a
# malformed one is a named 400 from the shared parser rather than a field
# H3 silently ignores.
CODE=$(curl -s -o "$TMP/b5b.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/video/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"x\",\"lora_paths\":\"not-an-array\"}")
if [ "$CODE" = "400" ] && grep -q "lora_paths" "$TMP/b5b.json"; then
  echo "PASS: malformed lora_paths on H3 is a named 400 (shared parser reached)"
else
  echo "FAIL: expected the lora_paths 400 on H3, got $CODE: $(cat "$TMP/b5b.json" | head -c 200)"; rc=1
fi
# A relative path must never reach mlx's loader (an MLX error kills the
# process, the BadLoraPath class) — it is a 400 naming the requirement.
CODE=$(curl -s -o "$TMP/b5c.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/video/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"x\",\"lora_paths\":[\"rel/style.safetensors\"]}")
if [ "$CODE" = "400" ] && grep -q "absolute" "$TMP/b5c.json"; then
  echo "PASS: a relative LoRA path on H3 is a named 400, never handed to mlx"
else
  echo "FAIL: expected the absolute-path 400 on H3, got $CODE: $(cat "$TMP/b5c.json" | head -c 200)"; rc=1
fi

# ── [4] Chained-window request surface (same server; both 400s pre-load).
CODE=$(curl -s -o "$TMP/b6.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/video/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"x\",\"chain_windows\":7}")
if [ "$CODE" = "400" ] && grep -q "chain_windows must be 1-6" "$TMP/b6.json"; then
  echo "PASS: out-of-range chain_windows is a named 400"
else
  echo "FAIL: expected the chain_windows-range 400, got $CODE: $(cat "$TMP/b6.json" | head -c 200)"; rc=1
fi
# A REF2VA pack cannot chain (no keyframe row to chain through) — refused by
# name, never a generation that silently ignores the request.
REFDIR="$TMP/models/fake/MiniMax-H3-Ref"
mkdir -p "$REFDIR"
for f in text_encoder transformer video_vae audio_vae; do
  dd if=/dev/zero of="$REFDIR/$f.safetensors" bs=1 count=0 seek=1000000 2>/dev/null
done
cat >"$REFDIR/config.json" <<'JSON'
{"model_type":"minimax_h3","partition":"ref2va","tasks":["t2va","ref2va"],"fps":24}
JSON
stop
boot 30GB "$TMP/turbo2.log" || exit 1
CODE=$(curl -s -o "$TMP/b7.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/video/generations" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"fake/MiniMax-H3-Ref\",\"prompt\":\"x\",\"chain_windows\":2}")
if [ "$CODE" = "400" ] && grep -q "REF2VA pack cannot serve" "$TMP/b7.json"; then
  echo "PASS: chaining on a REF2VA pack is a named 400"
else
  echo "FAIL: expected the ref2va-chain 400, got $CODE: $(cat "$TMP/b7.json" | head -c 200)"; rc=1
fi
stop

[ $rc -eq 0 ] && echo "ALL PASS" || echo "SOME FAILURES"
exit $rc
