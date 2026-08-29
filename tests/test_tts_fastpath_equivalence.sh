#!/usr/bin/env bash
# TTS fast-path equivalence guard, two contracts:
#   BYTE-IDENTICAL set (GPU-chained lazy predictor + fused SwiGLU / QK-norm+
#   RoPE / residual+RMSNorm): boot B (banded levers killed) must produce wavs
#   byte-identical to boot C (everything killed), and each lever must ENGAGE
#   on its boot while the kill boots show NO engagement (expectNoSpec class,
#   both directions).
#   BANDED set (KV-cached predictor steps / compiled chain — reduction-order
#   deviations): boot A (full default) must stay within speech bands of C
#   (duration ratio, energy, non-silence) — near-tie flips may change codes.
# Usage: TTS_MODEL=<dir> ./tests/test_tts_fastpath_equivalence.sh [port]
set -euo pipefail
PORT="${1:-11379}"
MODEL="${TTS_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3-TTS-12Hz-*-Base-* 2>/dev/null | head -1 || true)}"
[ -n "$MODEL" ] || { echo "SKIP: no qwen3_tts model (set TTS_MODEL)"; exit 0; }
[ -d "$MODEL" ] || { echo "FAIL: TTS_MODEL points at a missing dir: $MODEL" >&2; exit 1; }
BIN="${BIN:-./zig-out/bin/mlx-serve}"
TMP=$(mktemp -d /tmp/tts_fastpath.XXXXXX)
SRV=""
trap 'kill $SRV 2>/dev/null || true; rm -rf "$TMP"' EXIT

python3 - "$TMP" <<'PY'
import sys, wave, math, struct, json, base64, os
tmp = sys.argv[1]
frames = bytearray()
for i in range(24000):
    t = i / 24000.0
    env = 0.5 + 0.5 * math.sin(2 * math.pi * 3.0 * t)
    x = env * (0.3 * math.sin(2 * math.pi * 160.0 * t) + 0.15 * math.sin(2 * math.pi * 320.0 * t))
    frames += struct.pack('<h', int(x * 32767))
w = wave.open(os.path.join(tmp, 'ref.wav'), 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
w.writeframes(bytes(frames)); w.close()
json.dump({'input': 'The fused path must match the composed path byte for byte.'},
          open(os.path.join(tmp, 'req_plain.json'), 'w'))
b64 = base64.b64encode(open(os.path.join(tmp, 'ref.wav'), 'rb').read()).decode()
json.dump({'input': 'The fused path must match the composed path byte for byte.', 'ref_audio': b64},
          open(os.path.join(tmp, 'req_clone.json'), 'w'))
PY

wait_up() { # $1 = log
  for i in $(seq 1 90); do grep -q "Server listening" "$1" && return 0; sleep 1; done
  echo "FAIL: server did not start"; tail -30 "$1"; exit 1
}
speak() { # $1 = request json, $2 = out wav
  local code
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/audio/speech" -H 'Content-Type: application/json' \
    --data @"$1" -o "$2" -w "%{http_code}")
  [ "$code" = "200" ] || { echo "FAIL: http $code for $1"; exit 1; }
  [ "$(head -c 4 "$2")" = "RIFF" ] || { echo "FAIL: not a WAV ($1)"; exit 1; }
}

# ── Boot A: full default (KV-cached predictor — the banded arm) ──
LOG_A="$TMP/server_fast.log"
"$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG_A" 2>&1 &
SRV=$!
wait_up "$LOG_A"
speak "$TMP/req_plain.json" "$TMP/fast_plain.wav"
speak "$TMP/req_clone.json" "$TMP/fast_clone.wav"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; SRV=""

grep -q "GPU-chained code predictor engaged" "$LOG_A" || { echo "FAIL: GPU predictor never engaged"; exit 1; }
grep -q "KV-cached predictor steps engaged" "$LOG_A" || { echo "FAIL: KV-cached predictor never engaged"; exit 1; }
grep -q "fused SwiGLU kernel engaged" "$LOG_A" || { echo "FAIL: fused SwiGLU never engaged"; exit 1; }
grep -q "fused QK-norm+RoPE engaged" "$LOG_A" || { echo "FAIL: fused QK-norm+RoPE never engaged"; exit 1; }
grep -q "fused residual+RMSNorm kernel engaged" "$LOG_A" || { echo "FAIL: fused residual+RMSNorm never engaged"; exit 1; }
echo "PASS: [1/4] all levers engaged on the default boot"

# ── Boot B: banded levers killed, bit-exact levers on ──
LOG_B="$TMP/server_lazy.log"
MLX_SERVE_TTS_CP_CACHE=0 MLX_SERVE_TTS_COMPILE=0 \
  "$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG_B" 2>&1 &
SRV=$!
wait_up "$LOG_B"
speak "$TMP/req_plain.json" "$TMP/lazy_plain.wav"
speak "$TMP/req_clone.json" "$TMP/lazy_clone.wav"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; SRV=""
if grep -qE "KV-cached predictor steps engaged|compiled predictor chain engaged" "$LOG_B"; then
  echo "FAIL: a banded lever engaged with its kill switch set"; exit 1
fi

# ── Boot C: everything killed (the composed reference) ──
LOG_C="$TMP/server_composed.log"
MLX_SERVE_TTS_CP_CACHE=0 MLX_SERVE_TTS_COMPILE=0 MLX_SERVE_TTS_GPU_PREDICT=0 \
  MLX_SERVE_SWIGLU_FUSED=0 MLX_SERVE_QK_NORM_ROPE_FUSED=0 MLX_SERVE_TTS_ADD_RMSNORM=0 \
  "$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG_C" 2>&1 &
SRV=$!
wait_up "$LOG_C"
speak "$TMP/req_plain.json" "$TMP/composed_plain.wav"
speak "$TMP/req_clone.json" "$TMP/composed_clone.wav"
kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; SRV=""

if grep -qE "GPU-chained code predictor engaged|KV-cached predictor|compiled predictor chain|fused SwiGLU kernel engaged|fused QK-norm\+RoPE engaged|fused residual\+RMSNorm kernel engaged" "$LOG_C"; then
  echo "FAIL: a lever engaged on the everything-killed boot"; exit 1
fi
echo "PASS: [2/4] kill switches keep every lever off"

cmp -s "$TMP/lazy_plain.wav" "$TMP/composed_plain.wav" || { echo "FAIL: plain-voice WAV differs lazy-chain vs composed"; exit 1; }
cmp -s "$TMP/lazy_clone.wav" "$TMP/composed_clone.wav" || { echo "FAIL: cloned-voice WAV differs lazy-chain vs composed"; exit 1; }
echo "PASS: [3/4] bit-exact set byte-identical to composed path (plain + clone)"

# Banded arm: same speech within bands, never silence. The energy band applies
# to the CLONE pair only — a conditioned voice renders at stable energy
# (measured ratio ~1.0), while plain voice on a -Base checkpoint is a random
# unconditioned voice whose energy legitimately varies rendition to rendition.
python3 - "$TMP/fast_plain.wav" "$TMP/composed_plain.wav" "$TMP/fast_clone.wav" "$TMP/composed_clone.wav" <<'PY'
import sys, wave, array, math
def stats(p):
    w = wave.open(p); frames = w.readframes(w.getnframes())
    xs = array.array('h', frames)
    rms = math.sqrt(sum(x * x for x in xs) / max(1, len(xs)))
    return w.getnframes(), rms
for i, (a, b) in enumerate(((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4]))):
    na, ra = stats(a); nb, rb = stats(b)
    dr = na / nb
    assert 0.7 < dr < 1.4, f"duration ratio {dr:.2f} out of band ({a} vs {b})"
    assert ra > 200 and rb > 200, f"near-silent output ({ra}, {rb})"
    if i == 1:  # clone pair
        er = ra / rb
        assert 0.5 < er < 2.0, f"energy ratio {er:.2f} out of band"
print("PASS: [4/4] banded arm within duration/energy bands of composed")
PY
