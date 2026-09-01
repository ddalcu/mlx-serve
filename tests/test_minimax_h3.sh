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

# Which PARTITION this pack is. The two ship identical files and geometry —
# only the DiT weights and the declared task list differ — so the pack's own
# config is the only honest discriminator, and asserting fl2va adherence on a
# REF2VA DiT (or references on an FL2VA one) would be a checkpoint expectation
# in a server test's clothes.
TASKS=$(python3 - "$MODEL/config.json" <<'PY'
import json, sys
try:
    print(",".join(json.load(open(sys.argv[1])).get("tasks", [])))
except Exception:
    print("")
PY
)
case ",$TASKS," in (*,fl2va,*) HAS_FL2VA=yes;; (*) HAS_FL2VA=no;; esac
case ",$TASKS," in (*,ref2va,*) HAS_REF2VA=yes;; (*) HAS_REF2VA=no;; esac
echo "pack tasks: [$TASKS]"

# [1] capability + staged preflight engagement
curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q '"video"' \
  && echo "PASS: /v1/models advertises video" \
  || { echo "FAIL: /v1/models missing video capability"; rc=1; }
# The NUMBER is the assertion, not the line: the line printed happily while
# the stub's modality-static model_type routed the bill to the 64.5 GB sum.
# The three stages are DISJOINT (TE freed before the DiT loads, DiT freed
# before the VAEs), and the DiT sheds its AdaLN weights, so the 8-bit pack's
# bill is max(TE 26.28, DiT 32.83x0.65 + turbo 0.73 + 6 activations) = 28.06 —
# against a 64.5 GB sum and the 38.97 this used to read.
PEAK=$(grep -m1 "media peak" "$LOG" | sed -E 's/.*media peak ~([0-9.]+) GB.*/\1/')
if [ -n "$PEAK" ] && python3 -c "import sys; sys.exit(0 if 24.0 < float('$PEAK') < 32.0 else 1)"; then
  echo "PASS: media preflight billed the biggest stage (~$PEAK GB, not the ~64.5 sum)"
else
  echo "FAIL: media preflight peak '$PEAK' GB is not the staged bill"; rc=1
fi

# [2] the named-400 surface — all cheap, so they run before any generation
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"x","lora_path":"/tmp/nope.safetensors"}' -o /tmp/h3_lora.json -w "%{http_code}")
# H3 takes stacked LoRAs since 4ffc69b, so the assertion is no longer "the
# backend cannot honor this field" — it is that an unusable PATH is a named 400
# that says which field, checked BEFORE the DiT loads (its own loadFile check
# is reached minutes too late for a 400). This arm had asserted the old wording
# and been failing since stacked LoRAs shipped.
if [ "$code" = "400" ] && grep -q "lora_path" /tmp/h3_lora.json; then
  echo "PASS: unreadable lora_path -> named 400 naming the field"
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

# The staged bill above is a claim about a runtime nobody can check unless each
# stage reports what it actually held — the DiT term was 12 GiB wrong for months
# with only the DiT's own line to catch it. Both stages log, and the DiT's must
# land BELOW its file size (precomputeAdaln frees the AdaLN weights).
DITR=$(grep -m1 "dit resident" "$LOG" | sed -E 's/.*dit resident: ([0-9.]+) GB.*/\1/')
if grep -q "encoder resident" "$LOG" && [ -n "$DITR" ]; then
  echo "PASS: both staged loads report their residency (dit ${DITR} GB)"
else
  echo "FAIL: a stage bill with no residency line is uncheckable"; rc=1
fi
DITFILE=$(python3 -c "import os;print(f\"{os.path.getsize('$MODEL/transformer.safetensors')/1024**3:.2f}\")" 2>/dev/null)
if [ -n "$DITR" ] && [ -n "$DITFILE" ] && python3 -c "import sys; sys.exit(0 if float('$DITR') < float('$DITFILE') else 1)"; then
  echo "PASS: DiT settles below its file size (${DITR} < ${DITFILE} GB) — the bill's 65% is real"
else
  echo "FAIL: DiT resident ${DITR} GB vs file ${DITFILE} GB — the shed AdaLN weights are still there"; rc=1
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

# [4b] opt-in per-step JPEG (issue #208): preview:true attaches a JPEG to
# Generating events; absent preview keeps the original {stage,step,total} shape
# (pinned above). Cached-velocity steps may omit the image; a real forward at
# 1 step / 64px always produces one.
SSE_PREV=/tmp/test_minimax_h3_sse_preview.txt
curl -sN --max-time 900 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red fox in snow","num_frames":40,"width":64,"height":64,"steps":1,"seed":1,"stream":true,"preview":true}' >"$SSE_PREV"
python3 - "$SSE_PREV" <<'PY'
import sys, json, base64
jpg = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data: "):
        continue
    ev = json.loads(line[6:])
    if ev.get("type") != "progress":
        continue
    if "preview" not in ev:
        continue
    raw = base64.b64decode(ev["preview"])
    assert raw[:2] == b"\xff\xd8", raw[:8]
    assert ev.get("mime") == "image/jpeg"
    assert ev.get("w") and ev.get("h")
    jpg += 1
assert jpg >= 1, "preview:true must attach at least one JPEG to a Generating event"
print(f"PASS: SSE preview -> {jpg} JPEG progress event(s)")
PY
[ $? -eq 0 ] || rc=1

# [5] fl2va first-frame conditioning: a pinned high-contrast left/right split
# image must survive into frame 0 (left dark, right bright) regardless of the
# prompt — a t2va run that silently ignored the image shows no such structure.
# 256px: at 64px the keyframe collapses to FOUR cond tokens (2x2 patched
# grid) and adherence is chance — measured live; 256px = 64 cond rows and the
# split reproduces near-exactly (frame0 left 16 / right 231 vs input 20/235).
if [ "$HAS_FL2VA" != "yes" ]; then
  echo "SKIP: fl2va keyframe adherence (this pack declares no 'fl2va' task)"
else
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
fi

# A garbage keyframe must be a NAMED 400, never a silent t2va (the a2vid rule).
# Server behaviour, so it holds on either partition.
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
# Only the fl2va arm sends "fast": false, so the no-engagement half of the
# assertion only means anything on a pack that ran it.
if [ "$HAS_FL2VA" = "yes" ]; then
  if grep -qE "step-cache reused|attn-broadcast reused" "$LOG"; then
    echo "FAIL: fast levers engaged on a 'fast': false request"; rc=1
  else
    echo "PASS: 'fast': false kept every forward dense"
  fi
fi
code=$(curl -s --max-time 900 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a slow pan over dunes","num_frames":5,"width":64,"height":64,"steps":12,"seed":2}' -o /dev/null -w "%{http_code}")
if [ "$code" = "200" ] && grep -q "step-cache reused" "$LOG" && grep -q "attn-broadcast reused" "$LOG"; then
  echo "PASS: default-fast engages both levers (log-counted)"
else
  echo "FAIL: default-fast did not engage (http $code)"; rc=1
fi

# [7] ref2va. The two partitions ship IDENTICAL files and geometry — only the
# DiT weights and the declared task list differ — so an FL2VA pack handed
# references would generate happily while ignoring every one of them. Which
# half of this section runs is decided by the pack's own config, never by its
# directory name.
if [ "$HAS_REF2VA" != "yes" ]; then
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    -d '{"prompt":"x","num_frames":5,"width":64,"height":64,"ref_images":["QQ=="]}' -o /tmp/h3_norefs.json -w "%{http_code}")
  if [ "$code" = "400" ] && grep -q "ref2va" /tmp/h3_norefs.json; then
    echo "PASS: references against a non-REF2VA pack -> named 400"
  else
    echo "FAIL: refs on an FL2VA pack returned $code ($(head -c 160 /tmp/h3_norefs.json))"; rc=1
  fi
  echo "SKIP: ref2va generation cases (this pack declares no 'ref2va' task)"
else
  # Fixtures: one reference image, a 5-frame clip (5 is already ON the 17k+5
  # ladder, so nothing is snapped away), and a short stereo WAV.
  python3 - /tmp/h3_ref <<'PY'
import struct, zlib, sys, math, wave
base = sys.argv[1]
def png(path, w, h, pix):
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes(pix(x, y))
    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    out += chunk(b"IEND", b"")
    open(path, "wb").write(out)

# A reference IMAGE at 128px: `match` sizing halves it onto the 64px canvas.
png(base + "_img.png", 128, 128, lambda x, y: (230 if x < 64 else 25, 40, 200 if y < 64 else 30))
# Frames that MOVE, so a collapsed or reversed temporal axis cannot pass.
for i in range(5):
    png(f"{base}_v{i}.png", 64, 64, lambda x, y, i=i: (255 if abs(x - (8 + 10 * i)) < 6 else 20, 20, 20))
# 0.5 s of 16 kHz stereo — resampled to the VAE's 32 kHz server-side.
w = wave.open(base + "_aud.wav", "wb")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(16000)
w.writeframes(b"".join(struct.pack("<hh", int(12000 * math.sin(t / 20.0)), int(9000 * math.sin(t / 30.0)))
                       for t in range(8000)))
w.close()
PY

  # The named-400 surface. Every arm is a refusal the client can act on: a
  # silently dropped reference generates a clip that just ignores the user.
  b64() { base64 < "$1" | tr -d '\n'; }
  IMGB=$(b64 /tmp/h3_ref_img.png)
  AUDB=$(b64 /tmp/h3_ref_aud.wav)
  python3 - "$IMGB" "$AUDB" <<'PY'
import json, sys
img, aud = sys.argv[1], sys.argv[2]
base = {"prompt": "x", "num_frames": 5, "width": 64, "height": 64, "steps": 1}
frames = [img] * 5
cases = {
    "over_images": {"ref_images": [img] * 10},
    "over_videos": {"ref_videos": [{"frames": frames}] * 4},
    "over_audios": {"ref_audios": [aud] * 4},
    "short_video": {"ref_videos": [{"frames": [img] * 4}]},
    "empty_video": {"ref_videos": [{"frames": []}]},
    "bad_image":   {"ref_images": ["bm90IGFuIGltYWdl"]},
    "bad_audio":   {"ref_audios": ["bm90IGF1ZGlv"]},
    "bad_sizing":  {"ref_images": [img], "ref_image_size": "huge"},
    # Every per-type cap holds (9 + 3 + 3) and the SET is still over: MiniMax's
    # limit is 12 files across all three lists, which no per-type check can see.
    "over_total":  {"ref_images": [img] * 9,
                    "ref_videos": [{"frames": frames}] * 3,
                    "ref_audios": [aud] * 3},
}
for name, extra in cases.items():
    json.dump({**base, **extra}, open(f"/tmp/h3_ref_400_{name}.json", "w"))
PY
  for case in over_images:"at most 9" over_videos:"at most 3" over_audios:"at most 3" \
              short_video:"at least 5 frames" empty_video:"no frames" \
              bad_image:"ref_images" bad_audio:"ref_audios" bad_sizing:"ref_image_size" \
              over_total:"at most 12 files"; do
    name="${case%%:*}"; want="${case#*:}"; want="${want%\"}"; want="${want#\"}"
    code=$(curl -s --max-time 120 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
      --data @/tmp/h3_ref_400_"$name".json -o /tmp/h3_ref_400_"$name"_resp.json -w "%{http_code}")
    if [ "$code" = "400" ] && grep -qF "$want" /tmp/h3_ref_400_"$name"_resp.json; then
      echo "PASS: ref2va $name -> named 400"
    else
      echo "FAIL: ref2va $name returned $code ($(head -c 160 /tmp/h3_ref_400_"$name"_resp.json))"; rc=1
    fi
  done

  # A mixed reference set through the whole path: image + soundtracked clip +
  # standalone audio. The ORDINALS are a contract with the checkpoint (the
  # prompt says <Picture 1> / <Video 1> / <Audio j>), and a soundtrack consumes
  # an <Audio> ordinal ahead of any standalone one — so this asserts the
  # SECOND standalone audio is <Audio 2>, which an ordering bug cannot fake.
  REFLOG_BEFORE=$(grep -c "minimax-h3 reference" "$LOG" || true)
  python3 - "$IMGB" "$AUDB" /tmp/h3_ref <<'PY'
import base64, json, sys
img, aud, base = sys.argv[1], sys.argv[2], sys.argv[3]
frames = [base64.b64encode(open(f"{base}_v{i}.png", "rb").read()).decode() for i in range(5)]
json.dump({
    "prompt": "<Picture 1> the cat, <Video 1> the motion, <Audio 2> the mood",
    "num_frames": 5, "width": 64, "height": 64, "steps": 2, "seed": 11,
    "ref_images": [img],
    "ref_videos": [{"frames": frames, "audio": aud}],
    "ref_audios": [aud],
    "stream": True,
}, open("/tmp/h3_ref_mixed.json", "w"))
PY
  RSSE=/tmp/test_minimax_h3_ref_sse.txt
  curl -sN --max-time 1200 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    --data @/tmp/h3_ref_mixed.json >"$RSSE"
  python3 - "$RSSE" <<'PY'
import sys, json, base64
prog = 0; complete = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data: "): continue
    ev = json.loads(line[6:])
    if ev["type"] == "progress": prog += 1
    elif ev["type"] == "complete": complete = ev
assert complete is not None, "no complete event on the ref2va run"
assert prog >= 1, f"expected progress events, got {prog}"
F, H, W = complete["frames"], complete["height"], complete["width"]
assert (F, H, W) == (5, 64, 64), (F, H, W)
raw = base64.b64decode(complete["data"])
assert len(raw) == F * H * W * 3, f"rgb len {len(raw)} != {F*H*W*3}"
assert max(raw) - min(raw) > 20, "ref2va frames look uniform"
print(f"PASS: ref2va SSE -> {prog} progress events, {F}f {W}x{H} with a soundtrack")
PY
  [ $? -eq 0 ] || rc=1

  # Engagement, counted in the server's own log: the resolver's ORDER and
  # ORDINALS, and the layout actually growing reference rows. Output alone
  # cannot see a silently ignored reference set — the generation succeeds.
  python3 - "$LOG" <<'PY'
import re, sys
lines = [l for l in open(sys.argv[1]) if "minimax-h3 reference" in l]
tail = lines[-3:]
assert len(tail) == 3, f"expected 3 reference lines, got {len(lines)}"
kinds = [re.search(r"reference (\w+) #(\d+)", l).groups() for l in tail]
assert kinds == [("image", "1"), ("video", "1"), ("audio", "2")], kinds
cond = [l for l in open(sys.argv[1]) if "reference(s) encoded" in l][-1]
m = re.search(r"-> (\d+) visual / (\d+) audio cond rows", cond)
assert m, cond
vis, aud = int(m.group(1)), int(m.group(2))
# image (1 latent frame) + clip (2 latent frames) at a 4x4 latent grid, patch
# 2x2 => 4 rows per frame; the soundtrack and the standalone audio each pack
# 2 rows per latent frame.
assert vis == 12, f"expected 12 visual cond rows (1+2 latent frames x 4), got {vis}"
assert aud > 0 and aud % 4 == 0, f"audio cond rows {aud} are not two equal-length stereo blocks"
print(f"PASS: ref2va engagement — image#1/video#1/audio#2 in order, {vis} visual + {aud} audio cond rows")
PY
  [ $? -eq 0 ] || rc=1
fi

if [ $rc -eq 0 ]; then echo "ALL PASS"; else echo "FAILURES"; fi
exit $rc
