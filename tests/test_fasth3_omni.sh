#!/usr/bin/env bash
# FastH3 Omni (FastVideo's 8-step PDD student of MiniMax-H3 REF2VA) over HTTP.
#
# Pins what the pack's contract changes on the H3 request surface: the step
# count is the trained one (omitted or 8; anything else is a named 400), Turbo
# is refused, a request needs an image or video reference, and one small
# generation runs the PDD sampler and the trained sparse attention — their
# engagement lines, exactly 8 forwards and no fast-recipe reuse (a cached
# velocity is meaningless across fused blocks).
#
# Usage: [OMNI_MODEL=<pack dir>] ./tests/test_fasth3_omni.sh [port]   # SKIPs without the pack
#
# Hermetic twins: the `PDD` tests in plugins/minimax_h3/h3.zig, `zig build test -Dtest-filter="h3 vsa:"` (H3_VSA_FIXTURE from
# tests/dump_h3_vsa_fixtures.py), `h3 vsa kernel` (GPU; H3_VSA_UBENCH=1), tests/test_convert_minimax_h3_omni.py.
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11363}"
MODEL="${OMNI_MODEL:-$HOME/.mlx-serve/models/ddalcu/FastH3-Omni-MLX-Serve-8bit}"
[ -f "$MODEL/transformer.safetensors" ] || { echo "SKIP: no FastH3 Omni pack at $MODEL (set OMNI_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
WORK=$(mktemp -d)
LOG="$WORK/server.log"
MLX_SERVE_H3_RESIDENT=0 "$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
srv_wait_health "http://127.0.0.1:$PORT" 90 1 "$SRV" || kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server did not start"; tail -8 "$LOG"; exit 1; }
rc=0

python3 - "$WORK" <<'PY'
import base64, json, struct, sys, zlib
work = sys.argv[1]
def png(w, h, pix):
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes(pix(x, y))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
img = base64.b64encode(png(64, 64, lambda x, y: (230 if x < 32 else 25, 40, 200 if y < 32 else 30))).decode()
base = {"prompt": "<Picture 1> a red and blue toy spinning. overall_soundscape: a soft hum.",
        "num_frames": 5, "width": 64, "height": 64, "seed": 3}
cases = {
    "steps":   ({**base, "ref_images": [img], "steps": 4}, "exactly 8 steps"),
    "turbo":   ({**base, "ref_images": [img], "turbo": True}, "distilled student"),
    "norefs":  ({**base}, "at least one image or video"),
    "run":     ({**base, "ref_images": [img], "stream": True}, ""),
}
for name, (body, _) in cases.items():
    json.dump(body, open(f"{work}/{name}.json", "w"))
open(f"{work}/wants", "w").write("".join(f"{n}\t{w}\n" for n, (_, w) in cases.items() if w))
PY

while IFS=$'\t' read -r name want; do
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    --data @"$WORK/$name.json" -o "$WORK/$name.out" -w "%{http_code}")
  if [ "$code" = "400" ] && grep -qF "$want" "$WORK/$name.out"; then
    echo "PASS: $name -> named 400"
  else
    echo "FAIL: $name returned $code ($(head -c 160 "$WORK/$name.out"))"; rc=1
  fi
done <"$WORK/wants"

curl -sN --max-time 1800 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  --data @"$WORK/run.json" >"$WORK/run.sse"
python3 - "$WORK/run.sse" "$LOG" <<'PY'
import base64, json, re, sys
complete = None
for line in open(sys.argv[1]):
    if line.startswith("data: "):
        ev = json.loads(line[6:])
        if ev["type"] == "complete": complete = ev
assert complete is not None, "no complete event"
raw = base64.b64decode(complete["data"])
assert (complete["frames"], complete["height"], complete["width"]) == (5, 64, 64)
assert max(raw) - min(raw) > 20, "frames look uniform"
log = open(sys.argv[2]).read()
assert "PDD student: 8 fused blocks over 32 heads, max_t 0.999" in log, "PDD sampler not engaged"
assert len(re.findall(r"\[minimax-h3\] step \d+/8 sigma", log)) == 8, "expected exactly 8 forwards"
assert "reused" not in log, "fast recipe engaged on a distilled student"
assert re.search(r"\[minimax-h3\] VSA: \d+ dense \+ \d+ video tiles .*(nax|simd) arm", log), "VSA not engaged on a Metal kernel arm"
print("PASS: one Omni generation through the PDD sampler and VSA, 8 forwards, no reuse")
PY
[ $? -eq 0 ] || rc=1

if [ $rc -eq 0 ]; then echo "ALL PASS"; else echo "FAILURES"; tail -20 "$LOG"; fi
exit $rc
