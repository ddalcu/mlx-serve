#!/usr/bin/env bash
# MiniMax-H3 (Hailuo 3.0) text-to-audio-video endpoint test.
#
# Boots a server over the converted pack and pins the whole H3 request surface:
# video capability advertised, a small generation returns rgb8 frames + a
# pcm_s16le stereo track of the RIGHT lengths (frame count snapped to the
# model's 17k+5 ladder), the named-400 surface (LoRA is the one field the
# backend cannot honor in any form; a non-/32 canvas; chat against a video
# model), SSE progress -> complete, and the staged-residency media preflight
# engagement line (max(TE,DiT)+VAEs, never the 64.5 GB sum).
#
# Usage: [H3_MODEL=<dir>] ./tests/test_minimax_h3.sh [port]
set -uo pipefail
PORT="${1:-11361}"
MODEL="${H3_MODEL:-$HOME/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit}"
[ -f "$MODEL/transformer.safetensors" ] || { echo "SKIP: no MiniMax-H3 pack at $MODEL (set H3_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

LOG=/tmp/test_minimax_h3_server.log
"$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -8 "$LOG"; exit 1; }
  sleep 1
done
rc=0

# [1] capability + staged preflight engagement
curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q '"video"' \
  && echo "PASS: /v1/models advertises video" \
  || { echo "FAIL: /v1/models missing video capability"; rc=1; }
# The NUMBER is the assertion, not the line: the line printed happily while
# the stub's modality-static model_type routed the bill to the 64.5 GB sum.
PEAK=$(grep -m1 "media peak" "$LOG" | sed -E 's/.*media peak ~([0-9.]+) GB.*/\1/')
if [ -n "$PEAK" ] && python3 -c "import sys; sys.exit(0 if 30.0 < float('$PEAK') < 50.0 else 1)"; then
  echo "PASS: media preflight billed the staged peak (~$PEAK GB, not the ~64.5 sum)"
else
  echo "FAIL: media preflight peak '$PEAK' GB is not the staged max(TE,DiT)+VAEs"; rc=1
fi

# [2] the named-400 surface — all cheap, so they run before any generation
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"x","lora_path":"/tmp/nope.safetensors"}' -o /tmp/h3_lora.json -w "%{http_code}")
if [ "$code" = "400" ] && grep -q "LoRA" /tmp/h3_lora.json; then
  echo "PASS: lora_path -> named 400"
else
  echo "FAIL: lora_path returned $code ($(head -c 120 /tmp/h3_lora.json))"; rc=1
fi

code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"x","width":100,"height":64}' -o /tmp/h3_dims.json -w "%{http_code}")
if [ "$code" = "400" ] && grep -q "multiples of 32" /tmp/h3_dims.json; then
  echo "PASS: non-/32 canvas -> named 400"
else
  echo "FAIL: non-/32 canvas returned $code ($(head -c 120 /tmp/h3_dims.json))"; rc=1
fi

code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}' -o /tmp/h3_chat.json -w "%{http_code}")
if [ "$code" = "400" ] && grep -q "video generation model" /tmp/h3_chat.json; then
  echo "PASS: chat against H3 -> named 400 (video-modality message)"
else
  echo "FAIL: chat returned $code ($(head -c 160 /tmp/h3_chat.json))"; rc=1
fi

# [3] small non-stream generation: rgb8 + pcm_s16le with the RIGHT lengths.
# 5 frames sits on the 17k+5 ladder already; 64x64 keeps the DiT sequence tiny
# (the wall clock is the staged TE+DiT weight load, not the steps).
OUT=/tmp/test_minimax_h3.json
code=$(curl -s --max-time 900 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a calico cat blinking on a sunlit windowsill. overall_soundscape: soft rain.","num_frames":5,"width":64,"height":64,"steps":2,"seed":7}' \
  -o "$OUT" -w "%{http_code}")
if [ "$code" != "200" ]; then
  echo "FAIL: generation http $code"; head -c 300 "$OUT"; rc=1
else
  python3 - "$OUT" <<'PY'
import sys, json, base64
d = json.load(open(sys.argv[1]))
assert d["format"] == "rgb8", d.get("format")
assert d["fps"] == 24, d.get("fps")
F, H, W = d["frames"], d["height"], d["width"]
assert F == 5, f"requested 5 frames (on the ladder), got {F}"
assert (H, W) == (64, 64), (H, W)
raw = base64.b64decode(d["data"])
assert len(raw) == F * H * W * 3, f"rgb len {len(raw)} != {F*H*W*3}"
lo, hi = min(raw), max(raw)
assert hi - lo > 20, f"frames look uniform ({lo}..{hi})"
assert d.get("audio_format") == "pcm_s16le", d.get("audio_format")
assert d.get("audio_channels") == 2, d.get("audio_channels")
sr = d["audio_sample_rate"]
pcm = base64.b64decode(d["audio_data"])
n_frames_per_ch = len(pcm) // (2 * 2)
adur, vdur = n_frames_per_ch / sr, F / 24.0
# The audio VAE decodes whole latent windows; allow one 40 Hz latent hop of slack.
assert abs(adur - vdur) < 0.06, f"audio {adur:.3f}s vs video {vdur:.3f}s"
print(f"PASS: generation -> {F}f {W}x{H} rgb8 range {lo}..{hi}, audio {adur:.3f}s @{sr}Hz stereo")
PY
  [ $? -eq 0 ] || rc=1
fi

# [4] frame-count snapping is honest: 40 requested must come back 56 (17k+5),
# checked on the SSE path together with progress -> complete ordering.
SSE=/tmp/test_minimax_h3_sse.txt
curl -sN --max-time 900 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red fox in snow","num_frames":40,"width":64,"height":64,"steps":1,"seed":1,"stream":true}' >"$SSE"
python3 - "$SSE" <<'PY'
import sys, json, base64
prog = 0
complete = None
saw_complete_after_progress = False
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data: "):
        continue
    ev = json.loads(line[6:])
    if ev["type"] == "progress":
        assert complete is None, "progress after complete"
        prog += 1
        assert {"stage", "step", "total"} <= set(ev), ev
    elif ev["type"] == "complete":
        complete = ev
        saw_complete_after_progress = prog > 0
assert prog >= 2, f"expected progress events, got {prog}"
assert complete is not None, "no complete event"
assert saw_complete_after_progress, "complete arrived before any progress"
assert complete["frames"] == 56, f"40 requested must snap UP to 56, got {complete['frames']}"
raw = base64.b64decode(complete["data"])
assert len(raw) == complete["frames"] * complete["height"] * complete["width"] * 3
print(f"PASS: SSE -> {prog} progress events, complete with {complete['frames']} frames (40 snapped to 56)")
PY
[ $? -eq 0 ] || rc=1

# [5] fl2va first-frame conditioning: a pinned high-contrast left/right split
# image must survive into frame 0 (left dark, right bright) regardless of the
# prompt — a t2va run that silently ignored the image shows no such structure.
# 256px: at 64px the keyframe collapses to FOUR cond tokens (2x2 patched
# grid) and adherence is chance — measured live; 256px = 64 cond rows and the
# split reproduces near-exactly (frame0 left 16 / right 231 vs input 20/235).
IMG=/tmp/test_h3_first_frame.png
python3 - "$IMG" <<'PY'
import sys, struct, zlib
W, H = 256, 256
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
raw = bytearray()
for y in range(H):
    raw.append(0)
    for x in range(W):
        v = 20 if x < W // 2 else 235
        raw += bytes((v, v, v))
png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
png += chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
PY
B64=$(base64 < "$IMG" | tr -d '\n')
python3 -c "import json,sys;json.dump({'prompt':'a static abstract scene of two solid color fields','num_frames':5,'width':256,'height':256,'steps':12,'seed':3,'fast':False,'first_frame_image':sys.argv[1]}, open('/tmp/test_h3_fl2va_req.json','w'))" "$B64"
FL=/tmp/test_h3_fl2va.json
code=$(curl -s --max-time 900 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  --data @/tmp/test_h3_fl2va_req.json -o "$FL" -w "%{http_code}")
if [ "$code" != "200" ]; then
  echo "FAIL: fl2va http $code"; head -c 300 "$FL"; rc=1
else
  if grep -q "keyframe conditioning engaged" "$LOG"; then
    echo "PASS: fl2va keyframe engagement (server log)"
  else
    echo "FAIL: no keyframe engagement line — silent t2va fallback?"; rc=1
  fi
  python3 - "$FL" <<'PY'
import sys, json, base64
d = json.load(open(sys.argv[1]))
F, H, W = d["frames"], d["height"], d["width"]
raw = base64.b64decode(d["data"])
f0 = raw[:H * W * 3]
left = right = 0.0
for y in range(H):
    for x in range(W):
        g = sum(f0[(y * W + x) * 3:(y * W + x) * 3 + 3]) / 3.0
        if x < W // 2: left += g
        else: right += g
n = H * W / 2
lm, rm = left / n, right / n
print(f"fl2va frame0 left_mean={lm:.1f} right_mean={rm:.1f}")
assert rm - lm > 100, f"frame 0 did not adhere to the first-frame image (left {lm:.1f} vs right {rm:.1f})"
print("PASS: fl2va frame 0 adheres to the conditioning image")
PY
  [ $? -eq 0 ] || rc=1
fi

# A garbage keyframe must be a NAMED 400, never a silent t2va (the a2vid rule).
python3 -c "import json;json.dump({'prompt':'x','num_frames':5,'width':64,'height':64,'first_frame_image':'bm90IGFuIGltYWdl'}, open('/tmp/test_h3_badkf.json','w'))"
code=$(curl -s --max-time 60 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  --data @/tmp/test_h3_badkf.json -o /tmp/h3_badkf_resp.json -w "%{http_code}")
if [ "$code" = "400" ] && grep -qi "keyframe" /tmp/h3_badkf_resp.json; then
  echo "PASS: undecodable keyframe -> named 400"
else
  echo "FAIL: bad keyframe returned $code ($(head -c 120 /tmp/h3_badkf_resp.json))"; rc=1
fi

# [6] the fast recipe is DEFAULT-ON and must ENGAGE (counted, never inferred
# from output): a 12-step default-fast gen logs both reuse lines; the fl2va
# case above ran "fast": false and must NOT have engaged before this point.
if grep -qE "step-cache reused|attn-broadcast reused" "$LOG"; then
  echo "FAIL: fast levers engaged on a 'fast': false request"; rc=1
else
  echo "PASS: 'fast': false kept every forward dense"
fi
code=$(curl -s --max-time 900 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a slow pan over dunes","num_frames":5,"width":64,"height":64,"steps":12,"seed":2}' -o /dev/null -w "%{http_code}")
if [ "$code" = "200" ] && grep -q "step-cache reused" "$LOG" && grep -q "attn-broadcast reused" "$LOG"; then
  echo "PASS: default-fast engages both levers (log-counted)"
else
  echo "FAIL: default-fast did not engage (http $code)"; rc=1
fi

if [ $rc -eq 0 ]; then echo "ALL PASS"; else echo "FAILURES"; fi
exit $rc
