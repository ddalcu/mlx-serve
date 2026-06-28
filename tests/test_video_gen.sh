#!/usr/bin/env bash
# Native LTX-Video 2.3 text-to-video endpoint smoke test.
# Usage: LTX_MODEL=<dir> [LTX_GEMMA_DIR=<dir>] ./tests/test_video_gen.sh [port]
set -uo pipefail
PORT="${1:-11331}"
MODEL="${LTX_MODEL:-$(ls -d ~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q4/snapshots/* 2>/dev/null | head -1)}"
[ -n "$MODEL" ] || { echo "SKIP: no LTX model (set LTX_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

"$BIN" --model "$MODEL" --serve --port "$PORT" >/tmp/test_video_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
# wait for /health (model load is heavy: transformer + connector + vae + gemma)
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -8 /tmp/test_video_server.log; exit 1; }
  sleep 2
done

# capabilities advertise "video"
curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q '"video"' || { echo "FAIL: /v1/models missing video capability"; exit 1; }

OUT=/tmp/test_video_gen.json
code=$(curl -s --max-time 600 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red fox running through a snowy forest","num_frames":9,"height":256,"width":384,"steps":4,"seed":42}' \
  -o "$OUT" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: http $code"; head -c 300 "$OUT"; exit 1; }

# decode b64 RGB frames, verify dims + that it is real content (not uniform/garbage)
python3 - "$OUT" <<'PY'
import sys, json, base64
d = json.load(open(sys.argv[1]))
assert d["format"] == "rgb8", d
F, H, W = d["frames"], d["height"], d["width"]
raw = base64.b64decode(d["data"])
assert len(raw) == F * H * W * 3, f"len {len(raw)} != {F*H*W*3}"
lo, hi = min(raw), max(raw)
assert hi - lo > 40, f"frames look uniform ({lo}..{hi}) — likely broken decode"
print(f"PASS: /v1/video/generations -> {F} frames {W}x{H}, {len(raw)} rgb bytes, range {lo}..{hi}")
PY
rc=$?

# SSE streaming: progress events must arrive, then a complete event with frames.
SSE=/tmp/test_video_sse.txt
curl -sN --max-time 600 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red fox","num_frames":9,"height":256,"width":384,"steps":2,"seed":1,"stream":true}' >"$SSE"
python3 - "$SSE" <<'PY'
import sys, json, base64
prog = 0; complete = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data: "): continue
    ev = json.loads(line[6:])
    if ev["type"] == "progress":
        prog += 1
        assert {"stage", "step", "total"} <= set(ev), ev
    elif ev["type"] == "complete":
        complete = ev
assert prog >= 3, f"expected several progress events, got {prog}"
assert complete is not None, "no complete event"
raw = base64.b64decode(complete["data"])
assert len(raw) == complete["frames"] * complete["height"] * complete["width"] * 3
print(f"PASS: SSE stream -> {prog} progress events + complete ({complete['frames']} frames)")
PY
[ $? -eq 0 ] || rc=1

# Optional: mux to mp4 if ffmpeg is present (proves a playable clip).
if [ $rc -eq 0 ] && command -v ffmpeg >/dev/null 2>&1; then
  python3 -c "import json,base64;d=json.load(open('$OUT'));open('/tmp/tvg.rgb','wb').write(base64.b64decode(d['data']));print(d['width'],d['height'],d['frames'])" >/tmp/tvg.dims
  read W H F < /tmp/tvg.dims
  ffmpeg -y -f rawvideo -pix_fmt rgb24 -s "${W}x${H}" -r 24 -i /tmp/tvg.rgb -frames:v "$F" \
    -c:v libx264 -pix_fmt yuv420p /tmp/test_video_gen.mp4 >/dev/null 2>&1 \
    && echo "PASS: muxed /tmp/test_video_gen.mp4"
fi
exit $rc
