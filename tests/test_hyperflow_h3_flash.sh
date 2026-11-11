#!/usr/bin/env bash
# HyperFlow H3 Flash (Video Rebirth's two-time few-step derivative of MiniMax-H3) over HTTP.
#
# Pins what a two-time pack changes on the H3 request surface: only the trained step counts sample (4, 6, 8;
# anything else is a named 400), Turbo and LoRAs are refused, and one small generation in the pack's own task
# (first/last keyframes for FL2VA, an image reference for REF2VA) runs the two-time sampler for exactly 6 forwards.
#
# Usage: FLASH_MODEL=<pack dir> ./tests/test_hyperflow_h3_flash.sh [port]   # SKIPs without the pack
#
# Hermetic twins: the `two-time` tests in plugins/minimax_h3/{h3,config}.zig; DiT parity vs stable-diffusion.cpp:
# `minimax h3 flash parity` (MINIMAX_H3_FLASH_FIXTURE from tests/dump_minimax_h3_flash_fixture.py).
source "$(dirname "$0")/_lib_models.sh"
set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11364}"
MODEL="${FLASH_MODEL:-$(find_model ddalcu/HyperFlow-H3-Flash-FL2VA-MLX-Serve-8bit || echo "$HOME/.mlx-serve/models/ddalcu/HyperFlow-H3-Flash-FL2VA-MLX-Serve-8bit")}"
[ -f "$MODEL/transformer.safetensors" ] || { echo "SKIP: no HyperFlow H3 Flash pack at $MODEL (set FLASH_MODEL)"; exit 0; }
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

python3 - "$WORK" "$MODEL" <<'PY'
import base64, json, struct, sys, zlib
work, model = sys.argv[1], sys.argv[2]
def png(w, h, pix):
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes(pix(x, y))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
red = base64.b64encode(png(64, 64, lambda x, y: (230 if x < 32 else 25, 40, 200 if y < 32 else 30))).decode()
blue = base64.b64encode(png(64, 64, lambda x, y: (25, 40 if x < 32 else 200, 230))).decode()
tasks = json.load(open(f"{model}/config.json"))["tasks"]
if "fl2va" in tasks:
    base = {"prompt": "a toy turns from red to blue. overall_soundscape: a soft hum.", "first_frame_image": red, "last_frame_image": blue}
else:
    base = {"prompt": "<Picture 1> a red and blue toy spinning. overall_soundscape: a soft hum.", "ref_images": [red]}
base.update({"num_frames": 5, "width": 64, "height": 64, "seed": 3})
cases = {
    "steps":  ({**base, "steps": 5}, "only the step counts it was trained for"),
    "turbo":  ({**base, "turbo": True}, "distilled student"),
    "lora":   ({**base, "lora_paths": [f"{model}/audio_vae.safetensors"]}, "a two-time pack has dropped and factored"),
    "run":    ({**base, "steps": 6, "stream": True}, ""),
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
assert "two-time DiT: 6 steps on its trained grid" in log, "two-time sampler not engaged"
assert len(re.findall(r"\[minimax-h3\] step \d+/6 sigma", log)) == 6, "expected exactly 6 forwards"
assert "step 2/6 sigma 0.9827" in log, "not on the trained 6-step grid"
print("PASS: one Flash generation through the two-time sampler, 6 forwards on the trained grid")
PY
[ $? -eq 0 ] || rc=1

if [ $rc -eq 0 ]; then echo "ALL PASS"; else echo "FAILURES"; tail -20 "$LOG"; fi
exit $rc
