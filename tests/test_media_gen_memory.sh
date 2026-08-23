#!/usr/bin/env bash
# Media-gen memory accounting: a generation must hand back every byte it took.
#
# The class: MageFlow's `sliceSeq` wrapped an `mlx_slice` handle in `contiguous`
# and never freed it, and a live slice pins its PARENT's buffer — so each DiT
# block retained the img+txt pair it was handed (~47 MB), 12 blocks x 4 steps =
# ~2.2 GB per megapixel, per generation, compounding and surviving unload.
#
# Two things this must do that a naive memory test does not:
#   * measure `mlx_active` from /props, NOT RSS — Metal buffers never appear in
#     RSS on Apple Silicon (it sat flat at 8.5 GB while 10 GB leaked);
#   * VARY the size-driving shape — a fixed-size replay cannot tell a leak from
#     ordinary size-keyed caching.
#
# Skips gracefully when the checkpoint isn't downloaded.
# Usage: MEDIA_MEM_MODEL=<dir> ./tests/test_media_gen_memory.sh [port]
set -uo pipefail
PORT="${1:-11413}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
MODEL="${MEDIA_MEM_MODEL:-$HOME/.mlx-serve/models/ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit}"
[ -f "$MODEL/model_index.json" ] || [ -f "$MODEL/config.json" ] \
  || { echo "SKIP: no media-gen checkpoint at $MODEL"; exit 0; }

# Headless boot over the discovery root, so load + unload are by model id and
# the idle baseline is a server with NOTHING resident.
MODEL_DIR="$(cd "$MODEL/../.." && pwd)"
MODEL_ID="$(basename "$(dirname "$MODEL")")/$(basename "$MODEL")"
LOG=/tmp/test_media_gen_memory_server.log
"$BIN" --serve --port "$PORT" --model-dir "$MODEL_DIR" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
  sleep 1
done

# Bytes MLX currently holds in live arrays. Freed buffers drop out of this
# immediately (they move to the allocator cache), so it is the leak signal.
active() {
  curl -s "http://127.0.0.1:$PORT/props" | python3 -c '
import json,sys
try: print(json.load(sys.stdin)["memory"]["active_bytes"])
except Exception: print(-1)'
}
gb() { python3 -c "print(f'{$1/1073741824:.2f}')"; }
gen() { # WxH -> "" on success, "ERR <code>" otherwise
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 900 \
    -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"a lighthouse\",\"size\":\"$1\",\"steps\":4,\"seed\":1}")
  [ "$code" = "200" ] || echo "ERR $code"
}
unload() {
  curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/unload-model" \
    -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL_ID\"}"
  sleep 3
}

# 256 MB. The class leaks ~2.2 GB per megapixel, so even the smallest size below
# overshoots this by 2x on its own — wide enough to absorb allocator noise
# without ever absorbing a real leak.
TOL=268435456

BASE=$(active)
[ "$BASE" -ge 0 ] || { echo "FAIL: /props did not report memory.active_bytes"; exit 1; }
echo "baseline (no model): $(gb "$BASE") GB"

# ── in-session: varying sizes must not ratchet the resident footprint ──
# Sizes differ in BOTH dimensions and in aspect, so nothing here is explained by
# a size-keyed cache reusing one shape.
r=$(gen 512x512); [ -z "$r" ] || { echo "FAIL: first generation -> $r"; tail -5 "$LOG"; exit 1; }
AFTER_FIRST=$(active)
for size in 768x768 640x896 512x512; do
  r=$(gen "$size"); [ -z "$r" ] || { echo "FAIL: generation at $size -> $r"; exit 1; }
done
AFTER_MANY=$(active)
GROWTH=$((AFTER_MANY - AFTER_FIRST))
echo "loaded after 1 gen: $(gb "$AFTER_FIRST") GB   after 4 gens: $(gb "$AFTER_MANY") GB   growth: $(gb "$GROWTH") GB"
[ "$GROWTH" -le "$TOL" ] || { echo "FAIL: resident memory grew $(gb "$GROWTH") GB across 3 more generations"; exit 1; }
echo "PASS: generations at varying sizes do not ratchet resident memory"

# ── across cycles: the post-unload figure must return to the baseline ──
FIRST_UNLOADED=""
for cycle in 1 2 3; do
  if [ "$cycle" -gt 1 ]; then
    for size in 512x512 768x768; do
      r=$(gen "$size"); [ -z "$r" ] || { echo "FAIL: cycle $cycle generation at $size -> $r"; exit 1; }
    done
  fi
  unload
  U=$(active)
  DELTA=$((U - BASE))
  echo "cycle $cycle unloaded: $(gb "$U") GB (baseline +$(gb "$DELTA") GB)"
  [ "$DELTA" -le "$TOL" ] || { echo "FAIL: cycle $cycle left $(gb "$DELTA") GB above the pre-load baseline"; exit 1; }
  [ -n "$FIRST_UNLOADED" ] || FIRST_UNLOADED=$U
  CREEP=$((U - FIRST_UNLOADED))
  [ "$CREEP" -le "$TOL" ] || { echo "FAIL: post-unload memory grew $(gb "$CREEP") GB by cycle $cycle"; exit 1; }
done
echo "PASS: three load/generate/unload cycles return to the baseline with no creep"

echo "ALL PASS: media-gen memory accounting ($MODEL_ID)"
