#!/usr/bin/env bash
# Latent oracle: same prompt, seed and steps, dense bf16 versus w8a8.
# SKIPs unless a dense Qwen-Image-2.1 checkpoint is present.
# Default-on bar, both 512 and 1024, 30 steps: final latent cosine >= 0.999
# and relative RMS <= 0.03; step-0 DiT velocity cosine >= 0.999 and relative
# RMS <= 0.02. This run is under that bar, so the server default stays off.
# The exit check is the regression floor under the measured numbers.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
MODEL="${QWEN_IMAGE_DENSE:-$HOME/.mlx-serve/models/local/Qwen-Image-2.1-dense}"
[ -x "$BIN" ] || { echo "FAIL: build first"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "SKIP: no dense Qwen-Image-2.1 ($MODEL)"; exit 0; }
PORT="${1:-18481}"
PROMPT="a red fox in the snow."
STEPS="${QWEN_ORACLE_STEPS:-30}"
SEED=3
OUT="$(mktemp -d)"
SRV=""
HELD=0
release() {
  if [ -n "${SRV:-}" ]; then
    kill $SRV 2>/dev/null || true
    wait $SRV 2>/dev/null || true
    SRV=""
  fi
  if [ "$HELD" = 1 ]; then
    "$LOCK" release pr486 || true
    HELD=0
  fi
}
trap release EXIT

LOCK=/Users/beam/llm/sushi/scripts/gpu-lock.sh
gen_arm() { # gen_arm <flag> <size> <dumpdir>
  local flag="$1" size="$2" dump="$3"
  mkdir -p "$dump"
  local log="$dump/server.log"
  "$LOCK" acquire pr486
  HELD=1
  MLX_SERVE_QWEN_LATENT_DIR="$dump" "$BIN" --serve "$flag" --model-dir "$OUT" --port "$PORT" >"$log" 2>&1 &
  SRV=$!
  local i
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; cat "$log"; exit 1; }
    sleep 1
  done
  curl -s "http://127.0.0.1:$PORT/v1/load-model" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\"}" >/dev/null
  local code
  code="$(curl -s -m 7200 "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$(basename "$MODEL")\",\"size\":\"${size}x${size}\",\"steps\":$STEPS,\"seed\":$SEED,\"guidance_scale\":1,\"prompt\":\"$PROMPT\"}" \
    -o "$dump/resp.json" -w '%{http_code}')"
  [ "$code" = 200 ] || { echo "FAIL: $flag $size http $code"; tail -40 "$log"; exit 1; }
  kill $SRV 2>/dev/null || true
  wait $SRV 2>/dev/null || true
  SRV=""
  "$LOCK" release pr486 || true
  HELD=0
}

cmp_pair() { # cmp_pair <size>
  local size="$1"
  python3 - "$OUT/bf16-$size" "$OUT/w8-$size" "$size" <<'PY'
import sys, numpy as np
bf, w8, size = sys.argv[1], sys.argv[2], sys.argv[3]
# Default-on bar. Missing it keeps w8a8 opt-in.
ON = {
    "dit_step0": (0.999, 0.02),
    "latent_final": (0.999, 0.03),
}
# Regression floor under the measured 30-step run.
FLOOR = {
    "noise": (0.999999, 1e-5),
    "dit_step0": (0.999, 0.04),
    "latent_step0": (0.999, 0.01),
    "latent_final": (0.995, 0.10),
}
def load(dir, name):
    shape = tuple(int(x) for x in open(f"{dir}/{name}.shape").read().split())
    a = np.fromfile(f"{dir}/{name}.f32", dtype=np.float32)
    if a.size != int(np.prod(shape)):
        raise SystemExit(f"FAIL: {name} size {a.size} != {shape}")
    return a.reshape(shape).astype(np.float64)
def stats(a, b):
    a, b = a.ravel(), b.ravel()
    nb = np.linalg.norm(b)
    cos = float(a.dot(b) / (np.linalg.norm(a) * nb))
    rel = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-12))
    ratio = float(np.linalg.norm(a) / nb)
    return cos, rel, ratio
ok = True
meets_on = True
for name, (cos_floor, rel_floor) in FLOOR.items():
    cos, rel, ratio = stats(load(w8, name), load(bf, name))
    print(f"[oracle] {size} {name}: cos={cos:.6f} rel_rms={rel:.6f} rms_ratio={ratio:.6f}")
    if not (cos >= cos_floor and rel <= rel_floor):
        print(f"FAIL: {size} {name} below regression floor cos>={cos_floor} rel_rms<={rel_floor}")
        ok = False
    if name in ON:
        cos_on, rel_on = ON[name]
        if not (cos >= cos_on and rel <= rel_on):
            meets_on = False
            print(f"[oracle] {size} {name} misses the default-on bar cos>={cos_on} rel_rms<={rel_on}")
print(f"[oracle] {size} default-on bar: {'MET' if meets_on else 'NOT MET'}")
raise SystemExit(0 if ok else 1)
PY
}

if [ -n "${QWEN_ORACLE_REUSE:-}" ]; then OUT="$QWEN_ORACLE_REUSE"; fi
for size in 512 1024; do
  if [ -z "${QWEN_ORACLE_REUSE:-}" ]; then
    gen_arm --no-w8a8 "$size" "$OUT/bf16-$size"
    gen_arm --w8a8 "$size" "$OUT/w8-$size"
  fi
  cmp_pair "$size" || FAIL=1
done
[ "${FAIL:-0}" = 0 ] && echo "ALL PASS" || exit 1
