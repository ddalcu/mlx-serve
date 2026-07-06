#!/usr/bin/env bash
# FLUX low-memory mode (phased text encoder) equivalence + engagement.
#
# Low-mem mode (`MLXSERVE_LOWMEM=1`, default-on in the iOS build) loads the
# FLUX text encoder lazily per request and FREES it right after the prompt
# encode — roughly halving the pipeline's resident bytes during the denoise
# loop (the iPhone jetsam fix). The conditioning tensor is materialized before
# the free, so outputs must be BYTE-IDENTICAL to the resident-encoder path.
#
# Asserts:
#   1. normal-mode PNG == low-mem PNG at the same seed (byte compare)
#   2. the low-mem path actually ENGAGED (log lines) — equality alone can't
#      catch a silent fallback to the resident path (spec-decode lesson)
#
# SKIPs without a FLUX model dir (default: the 3-bit klein).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-11321}"
MODEL_DIR="${FLUX_LOWMEM_MODEL:-$HOME/.mlx-serve/models/mlx-community/FLUX.2-Klein-4B-3bit}"
BIN="$ROOT/zig-out/bin/mlx-serve"
TMP="$(mktemp -d)"
trap 'kill $SRV_PID 2>/dev/null || true; rm -rf "$TMP"' EXIT

pass() { echo "  ✅ $1"; }
fail() { echo "  ❌ $1"; exit 1; }

[ -f "$MODEL_DIR/config.json" ] || { echo "SKIP: no FLUX model at $MODEL_DIR"; exit 0; }
[ -x "$BIN" ] || fail "build first: zig build -Doptimize=ReleaseFast"

REQ='{"model":"m","prompt":"a red fox sitting in fresh snow at golden hour, sharp detail","size":"512x512","steps":4,"seed":7}'

run_one() { # run_one <png-out> <log-out> [env...]
  local png="$1" log="$2"; shift 2
  env "$@" "$BIN" --serve --port "$PORT" --model-dir "$HOME/.mlx-serve/models" --log-level debug >"$log" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 30); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
  local model_id
  model_id="$(curl -fsS -X POST "http://127.0.0.1:$PORT/v1/load-model" -H 'Content-Type: application/json' \
    --max-time 300 -d "{\"model\":\"$MODEL_DIR\"}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['model']['id'])")" || fail "load-model failed"
  curl -fsS -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
    --max-time 590 -d "${REQ/\"m\"/\"$model_id\"}" \
    | python3 -c "import json,base64,sys; d=json.load(sys.stdin); open('$png','wb').write(base64.b64decode(d['data'][0]['b64_json']))" \
    || fail "generation failed"
  kill $SRV_PID 2>/dev/null || true
  wait $SRV_PID 2>/dev/null || true
}

echo "── normal mode (resident text encoder) ──"
run_one "$TMP/normal.png" "$TMP/normal.log"
pass "normal-mode PNG generated"

echo "── low-mem mode (phased text encoder) ──"
run_one "$TMP/lowmem.png" "$TMP/lowmem.log" MLXSERVE_LOWMEM=1
pass "low-mem PNG generated"

cmp "$TMP/normal.png" "$TMP/lowmem.png" || fail "outputs differ — low-mem must be byte-identical"
pass "byte-identical outputs"

grep -q "low-mem mode: text encoder loads per request" "$TMP/lowmem.log" || fail "low-mem load path did not engage"
grep -q "low-mem: text encoder freed after encode" "$TMP/lowmem.log" || fail "encoder free did not engage"
grep -q "low-mem" "$TMP/normal.log" && fail "normal mode unexpectedly took the low-mem path"
pass "low-mem engaged in low-mem mode only"

echo
echo "✅ FLUX low-mem equivalence test passed."
