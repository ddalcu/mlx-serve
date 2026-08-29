#!/usr/bin/env bash
# Native LTX-Video 2.3 text-to-video endpoint smoke test (incl. audio track).
# Usage: LTX_MODEL=<dir> [LTX_GEMMA_DIR=<dir>] [LTX_AUDIO_VAE=<file>] ./tests/test_video_gen.sh [port]
set -uo pipefail
PORT="${1:-11331}"
MODEL="${LTX_MODEL:-$(ls -d ~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-q4/snapshots/* 2>/dev/null | head -1)}"
[ -n "$MODEL" ] || { echo "SKIP: no LTX model (set LTX_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

# Audio is decoded only when the q4 audio weights (audio_vae.safetensors +
# vocoder.safetensors) sit in the model dir. When both are present the response
# MUST carry a non-silent stereo track synced to the video duration.
# Override the lookup dir with LTX_AUDIO_DIR if your audio files live elsewhere.
AUDIO_DIR="${LTX_AUDIO_DIR:-$MODEL}"
EXPECT_AUDIO=0
if [ -f "$AUDIO_DIR/audio_vae.safetensors" ] && [ -f "$AUDIO_DIR/vocoder.safetensors" ]; then
  export LTX_AUDIO_DIR="$AUDIO_DIR"; EXPECT_AUDIO=1; echo "audio VAE+vocoder found in $AUDIO_DIR -> expecting a sound track"
else
  echo "no audio VAE/vocoder in $AUDIO_DIR -> video-only run"
fi

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

# The pack's own config.json decides which text encoder loads, and a 2.5 pack
# ships one INSIDE itself. Getting this wrong does not crash — the connector
# would project a stack from the wrong encoder and generate a plausible video
# for a prompt nobody typed — so assert the resolution rather than the output.
VERSION=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('model_version','2.3.0'))" "$MODEL/config.json" 2>/dev/null || echo 2.3.0)
case "$VERSION" in
  2.5*)
    grep -q "LTX v25" /tmp/test_video_server.log || { echo "FAIL: 2.5 pack did not load as v25"; grep -i "\[video\]" /tmp/test_video_server.log | head -3; exit 1; }
    grep -q "gemma text encoder: $MODEL/gemma4-12b-ltx-v1" /tmp/test_video_server.log || { echo "FAIL: 2.5 pack did not resolve its in-pack text encoder"; grep -i "text encoder" /tmp/test_video_server.log | head -3; exit 1; }
    echo "PASS: LTX 2.5 resolved its own in-pack Gemma-4 text encoder"
    ;;
  *)
    grep -q "LTX v23" /tmp/test_video_server.log || { echo "FAIL: 2.3 pack did not load as v23"; grep -i "\[video\]" /tmp/test_video_server.log | head -3; exit 1; }
    ;;
esac

OUT=/tmp/test_video_gen.json
code=$(curl -s --max-time 600 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red fox running through a snowy forest","num_frames":9,"height":256,"width":384,"steps":4,"seed":42}' \
  -o "$OUT" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: http $code"; head -c 300 "$OUT"; exit 1; }

# decode b64 RGB frames, verify dims + that it is real content (not uniform/garbage),
# and (when the audio VAE is present) a non-silent stereo track synced to the clip.
python3 - "$OUT" "$EXPECT_AUDIO" <<'PY'
import sys, json, base64, struct, math, wave
d = json.load(open(sys.argv[1]))
expect_audio = sys.argv[2] == "1"
assert d["format"] == "rgb8", d
F, H, W = d["frames"], d["height"], d["width"]
raw = base64.b64decode(d["data"])
assert len(raw) == F * H * W * 3, f"len {len(raw)} != {F*H*W*3}"
lo, hi = min(raw), max(raw)
assert hi - lo > 40, f"frames look uniform ({lo}..{hi}) — likely broken decode"
print(f"PASS: /v1/video/generations -> {F} frames {W}x{H}, {len(raw)} rgb bytes, range {lo}..{hi}")

has_audio = "audio_data" in d
if expect_audio:
    assert has_audio, "audio VAE present but response has NO audio_data"
if has_audio:
    assert d["audio_format"] == "pcm_s16le", d["audio_format"]
    sr, ch = d["audio_sample_rate"], d["audio_channels"]
    assert ch == 2, f"expected stereo, got {ch}ch"
    pcm = base64.b64decode(d["audio_data"])
    n = len(pcm) // 2
    samples = struct.unpack("<%dh" % n, pcm)
    aframes = n // ch
    adur = aframes / sr
    vdur = F / d["fps"]
    peak = max(abs(s) for s in samples)
    rms = math.sqrt(sum(s * s for s in samples) / n)
    assert peak > 50, f"audio is silent (peak {peak})"
    # duration must track the video within ~150 ms (causal-crop slack)
    assert abs(adur - vdur) < 0.15, f"audio {adur:.3f}s vs video {vdur:.3f}s out of sync"
    w = wave.open("/tmp/test_video_gen.wav", "wb"); w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm); w.close()
    print(f"PASS: audio track {ch}ch {sr}Hz {adur:.3f}s (video {vdur:.3f}s) peak={peak} rms={rms:.0f} -> /tmp/test_video_gen.wav")
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

# ── Image-to-video (first-frame conditioning) ──────────────────────────────
# Only when the VAE encoder is present. A high-contrast left/right split image
# is pinned as the clean first frame; if conditioning works the decoded frame 0
# reconstructs the split (left dark, right bright) regardless of the prompt.
if [ -f "$MODEL/vae_encoder.safetensors" ]; then
  echo "vae_encoder.safetensors found -> testing image-to-video"
  IMG=/tmp/test_i2v_input.png
  python3 - "$IMG" <<'PY'
import sys, struct, zlib
W, H = 384, 256
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
raw = bytearray()
for y in range(H):
    raw.append(0)  # filter byte
    for x in range(W):
        v = 20 if x < W // 2 else 235   # left dark, right bright
        raw += bytes((v, v, v))
png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
png += chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
PY
  B64=$(base64 < "$IMG" | tr -d '\n')
  I2V=/tmp/test_video_i2v.json
  python3 -c "import json,sys;json.dump({'prompt':'a red fox running through a snowy forest','num_frames':9,'height':256,'width':384,'steps':4,'seed':7,'first_frame_image':sys.argv[1]}, open('/tmp/test_i2v_req.json','w'))" "$B64"
  code=$(curl -s --max-time 600 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    --data @/tmp/test_i2v_req.json -o "$I2V" -w "%{http_code}")
  if [ "$code" = "200" ]; then
    python3 - "$I2V" <<'PY'
import sys, json, base64
d = json.load(open(sys.argv[1]))
F, H, W = d["frames"], d["height"], d["width"]
raw = base64.b64decode(d["data"])
# frame 0, RGB → grayscale; compare left vs right half means.
f0 = raw[:H * W * 3]
def gray(px, i): return (px[i] + px[i+1] + px[i+2]) / 3.0
left = right = nl = nr = 0.0
for y in range(H):
    for x in range(W):
        g = gray(f0, (y * W + x) * 3)
        if x < W // 2: left += g; nl += 1
        else: right += g; nr += 1
lm, rm = left / nl, right / nr
print(f"I2V frame0 left_mean={lm:.1f} right_mean={rm:.1f}")
# The pinned first frame must reconstruct the split: right (235) clearly brighter
# than left (20). A t2v (ignored image) frame 0 would not show this structure.
assert rm - lm > 30, f"first frame did not adhere to conditioning image (left {lm:.1f} vs right {rm:.1f})"
print("PASS: image-to-video first frame adheres to the conditioning image")
PY
    [ $? -eq 0 ] || rc=1
  else
    echo "FAIL: I2V http $code"; head -c 300 "$I2V"; rc=1
  fi

  # ── First + last frame (#260) ─────────────────────────────────────────────
  # The last anchor is a top-dark/bottom-bright split; the first stays the
  # left/right one. Frame 0 must show the left/right split, the final frame
  # the top/bottom one, and the engine must log BOTH anchors engaged.
  LAST_IMG=/tmp/test_flf_last.png
  python3 - "$LAST_IMG" <<'PY'
import sys, struct, zlib
W, H = 384, 256
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
raw = bytearray()
for y in range(H):
    raw.append(0)
    v = 20 if y < H // 2 else 235   # top dark, bottom bright
    raw += bytes((v, v, v)) * W
png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
png += chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
PY
  LB64=$(base64 < "$LAST_IMG" | tr -d '\n')
  FLF=/tmp/test_video_flf.json
  python3 -c "import json,sys;json.dump({'prompt':'a red fox running through a snowy forest','num_frames':17,'height':256,'width':384,'steps':4,'seed':7,'first_frame_image':sys.argv[1],'last_frame_image':sys.argv[2]}, open('/tmp/test_flf_req.json','w'))" "$B64" "$LB64"
  code=$(curl -s --max-time 600 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    --data @/tmp/test_flf_req.json -o "$FLF" -w "%{http_code}")
  if [ "$code" = "200" ]; then
    grep -q "\[ltx\] keyframe conditioning: first=true last=true" /tmp/test_video_server.log \
      || { echo "FAIL: last-frame anchor not engaged (no first=true last=true log line)"; rc=1; }
    python3 - "$FLF" <<'PY'
import sys, json, base64
d = json.load(open(sys.argv[1]))
F, H, W = d["frames"], d["height"], d["width"]
assert F == 17, f"expected 17 frames, got {F}"
raw = base64.b64decode(d["data"])
fb = H * W * 3
def gray(px, i): return (px[i] + px[i+1] + px[i+2]) / 3.0
def halves(frame, axis):
    a = b = na = nb = 0.0
    for y in range(H):
        for x in range(W):
            g = gray(frame, (y * W + x) * 3)
            first = (x < W // 2) if axis == "x" else (y < H // 2)
            if first: a += g; na += 1
            else: b += g; nb += 1
    return a / na, b / nb
l, r = halves(raw[:fb], "x")
t, b = halves(raw[(F - 1) * fb:F * fb], "y")
print(f"FLF frame0 left={l:.1f} right={r:.1f}; last top={t:.1f} bottom={b:.1f}")
assert r - l > 30, f"first frame did not adhere to the first anchor ({l:.1f} vs {r:.1f})"
assert b - t > 30, f"last frame did not adhere to the last anchor ({t:.1f} vs {b:.1f})"
print("PASS: first/last frame anchors both reconstruct")
PY
    [ $? -eq 0 ] || rc=1
  else
    echo "FAIL: FLF http $code"; head -c 300 "$FLF"; rc=1
  fi
  # Refusals are named, never a silent text-to-video downgrade.
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    -d '{"prompt":"x","num_frames":9,"height":256,"width":384,"steps":2,"last_frame_image":"!!notbase64"}' -o /tmp/test_flf_bad.json -w "%{http_code}")
  [ "$code" = "400" ] && grep -q "last_frame_image" /tmp/test_flf_bad.json \
    && echo "PASS: bad last_frame_image -> named 400" \
    || { echo "FAIL: bad last_frame_image http $code"; head -c 200 /tmp/test_flf_bad.json; rc=1; }
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    --data "{\"prompt\":\"x\",\"num_frames\":5,\"height\":256,\"width\":384,\"steps\":2,\"last_frame_image\":\"$LB64\"}" -o /tmp/test_flf_short.json -w "%{http_code}")
  [ "$code" = "400" ] && grep -q "9 frames" /tmp/test_flf_short.json \
    && echo "PASS: last frame on a one-latent-frame canvas -> named 400" \
    || { echo "FAIL: short canvas with last frame http $code"; head -c 200 /tmp/test_flf_short.json; rc=1; }
else
  echo "no vae_encoder.safetensors in $MODEL -> skipping image-to-video test"
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    -d '{"prompt":"x","num_frames":9,"height":256,"width":384,"steps":2,"last_frame_image":"AAAA"}' -o /tmp/test_flf_noenc.json -w "%{http_code}")
  [ "$code" = "400" ] && grep -q "vae_encoder" /tmp/test_flf_noenc.json \
    && echo "PASS: last_frame_image without encoder -> named 400" \
    || { echo "FAIL: last_frame_image without encoder http $code"; rc=1; }
fi

# ── Two-stage pipeline (dev CFG half-res → x2 upsample → distilled refine) ──
# Needs BOTH transformer variants + the spatial upsampler + the VAE encoder
# (latent statistics). Small canvas keeps the guided stage affordable.
if [ -f "$MODEL/transformer-dev.safetensors" ] && [ -f "$MODEL/transformer-distilled.safetensors" ] \
   && [ -f "$MODEL/spatial_upscaler_x2_v1_1.safetensors" ] && [ -f "$MODEL/vae_encoder.safetensors" ]; then
  echo "two-stage prerequisites found -> testing pipeline=two_stage"
  TS=/tmp/test_video_two_stage.json
  code=$(curl -s --max-time 1200 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    -d '{"prompt":"a red fox running through a snowy forest","pipeline":"two_stage","num_frames":9,"height":256,"width":384,"steps":4,"seed":42}' \
    -o "$TS" -w "%{http_code}")
  if [ "$code" = "200" ]; then
    python3 - "$TS" <<'PY'
import sys, json, base64
d = json.load(open(sys.argv[1]))
F, H, W = d["frames"], d["height"], d["width"]
raw = base64.b64decode(d["data"])
assert len(raw) == F * H * W * 3, f"len {len(raw)} != {F*H*W*3}"
# two-stage refines at the FULL grid: 256x384 in → 256x384 out
assert (H, W) == (256, 384), f"unexpected output dims {W}x{H}"
lo, hi = min(raw), max(raw)
assert hi - lo > 40, f"two-stage frames look uniform ({lo}..{hi})"
print(f"PASS: two_stage -> {F} frames {W}x{H}, range {lo}..{hi}")
PY
    [ $? -eq 0 ] || rc=1
  else
    echo "FAIL: two_stage http $code"; head -c 300 "$TS"; rc=1
  fi
  # invalid grid (not divisible by 64) must 400, never silently downgrade
  code=$(curl -s --max-time 60 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    -d '{"prompt":"x","pipeline":"two_stage","num_frames":9,"height":224,"width":384,"steps":2,"seed":1}' \
    -o /dev/null -w "%{http_code}")
  if [ "$code" = "400" ]; then
    echo "PASS: two_stage rejects a non-/64 grid with 400"
  else
    echo "FAIL: two_stage non-/64 grid returned $code (want 400)"; rc=1
  fi
else
  echo "two-stage weights incomplete in $MODEL -> skipping two-stage test"
fi

# ── Audio-to-video (a2vid): user WAV frozen as the soundtrack ───────────────
# Needs the two-stage prerequisites + the audio VAE (encoder weights ride in
# audio_vae.safetensors). The ORIGINAL clip must come back in the response
# (native rate/channels, trimmed to the video duration); the ENGAGEMENT proof
# is the server-log line — response presence alone can't distinguish a silent
# fallback to generated audio (the spec-decode dispatch lesson).
if [ -f "$MODEL/transformer-dev.safetensors" ] && [ -f "$MODEL/transformer-distilled.safetensors" ] \
   && [ -f "$MODEL/spatial_upscaler_x2_v1_1.safetensors" ] && [ -f "$MODEL/vae_encoder.safetensors" ] \
   && [ "$EXPECT_AUDIO" = "1" ]; then
  echo "a2vid prerequisites found -> testing audio-to-video"
  WAV=/tmp/test_a2v_input.wav
  python3 - "$WAV" <<'PY'
import sys, wave, struct, math
sr = 44100  # NOT 16 kHz — exercises the server-side resample of the cond path
n = sr      # 1 s stereo, longer than the 9-frame clip so mux trimming shows
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        l = int(0.5 * 32767 * math.sin(2 * math.pi * 440 * t))
        r = int(0.5 * 32767 * math.sin(2 * math.pi * 660 * t))
        frames += struct.pack("<hh", l, r)
    w.writeframes(bytes(frames))
PY
  B64=$(base64 < "$WAV" | tr -d '\n')
  python3 -c "import json,sys;json.dump({'prompt':'a woman speaking to the camera in a bright kitchen, natural voice','pipeline':'two_stage','num_frames':9,'height':256,'width':384,'steps':4,'seed':11,'audio':sys.argv[1]}, open('/tmp/test_a2v_req.json','w'))" "$B64"
  A2V=/tmp/test_video_a2v.json
  code=$(curl -s --max-time 1200 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    --data @/tmp/test_a2v_req.json -o "$A2V" -w "%{http_code}")
  if [ "$code" = "200" ]; then
    python3 - "$A2V" <<'PY'
import sys, json, base64, struct, math
d = json.load(open(sys.argv[1]))
F, H, W = d["frames"], d["height"], d["width"]
raw = base64.b64decode(d["data"])
assert len(raw) == F * H * W * 3, f"len {len(raw)} != {F*H*W*3}"
assert "audio_data" in d, "a2vid response missing audio_data"
assert d["audio_sample_rate"] == 44100, f"expected ORIGINAL 44100 Hz passthrough, got {d['audio_sample_rate']}"
assert d["audio_channels"] == 2, d["audio_channels"]
pcm = base64.b64decode(d["audio_data"])
total = len(pcm) // 2  # s16 samples, interleaved
want_frames = int(F / 24.0 * 44100)
assert total == want_frames * 2, f"expected {want_frames} trimmed sample-frames, got {total//2}"
# The muxed track must be the ORIGINAL clip (440 Hz left channel), not a VAE
# re-synth: compare against the source sine (PCM16 round-trip tolerance).
bad = 0
for i in range(1000):
    l = struct.unpack_from("<h", pcm, i * 4)[0]
    ref = 0.5 * 32767 * math.sin(2 * math.pi * 440 * (i / 44100.0))
    if abs(l - ref) > 3: bad += 1
assert bad < 5, f"muxed audio deviates from the original clip ({bad}/1000 samples off)"
print(f"PASS: a2vid -> {F} frames + original 44.1 kHz clip trimmed to {total//2} sample-frames")
PY
    [ $? -eq 0 ] || rc=1
    if grep -q "audio-to-video:" /tmp/test_video_server.log; then
      echo "PASS: a2vid conditioning engaged (server log)"
    else
      echo "FAIL: no a2vid engagement line in server log (silent t2v fallback?)"; rc=1
    fi
  else
    echo "FAIL: a2vid http $code"; head -c 300 "$A2V"; rc=1
  fi
  # one-stage + audio must 400 — a2vid is two-stage only, never a silent ignore
  python3 -c "import json;r=json.load(open('/tmp/test_a2v_req.json'));r['pipeline']='one_stage';json.dump(r,open('/tmp/test_a2v_req_1s.json','w'))"
  code=$(curl -s --max-time 60 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    --data @/tmp/test_a2v_req_1s.json -o /dev/null -w "%{http_code}")
  if [ "$code" = "400" ]; then
    echo "PASS: a2vid rejects pipeline=one_stage with 400"
  else
    echo "FAIL: a2vid one_stage returned $code (want 400)"; rc=1
  fi
else
  echo "a2vid prerequisites incomplete in $MODEL -> skipping audio-to-video test"
fi

# ── DiffVAE decoder (`"decoder":"diffusion"`) ──────────────────────────────
# LTX's own diffusion decoder, which their published clips are decoded with.
# Only the 8-bit 2.5 pack ships `vae_diffusion_decoder.safetensors`; a pack
# without it must answer a NAMED 400, never silently fall back to the conv
# decoder (the two-stage-prerequisite precedent). ENGAGEMENT is the log line —
# a 200 alone cannot tell the two decoders apart, and per-frame pixel metrics
# fork with content, so this asserts the resolution and that the bytes DIFFER.
DV_OUT=/tmp/test_video_gen_diffvae.json
DV_REQ='{"prompt":"a red fox running through a snowy forest","num_frames":9,"height":256,"width":384,"steps":4,"seed":42'
code=$(curl -s --max-time 900 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d "$DV_REQ,\"decoder\":\"diffusion\"}" -o "$DV_OUT" -w "%{http_code}")
if [ -f "$MODEL/vae_diffusion_decoder.safetensors" ]; then
  if [ "$code" != "200" ]; then
    echo "FAIL: decoder=diffusion returned $code"; head -c 300 "$DV_OUT"; rc=1
  else
    grep -q "\[video\] decoder: diffusion" /tmp/test_video_server.log \
      || { echo "FAIL: decoder=diffusion did not engage (no log line)"; rc=1; }
    grep -q "\[diffvae\] NA kernel engaged" /tmp/test_video_server.log \
      || { echo "FAIL: the DiffVAE NA kernel never ran"; rc=1; }
    grep -q "\[diffvae\] decoding " /tmp/test_video_server.log \
      || { echo "FAIL: no DiffVAE decode line"; rc=1; }
    python3 - "$OUT" "$DV_OUT" <<'PYEOF' || rc=1
import base64, json, sys
conv = json.load(open(sys.argv[1]))
diff = json.load(open(sys.argv[2]))
if (conv["frames"], conv["height"], conv["width"]) != (diff["frames"], diff["height"], diff["width"]):
    print("FAIL: diffusion decode changed the clip geometry"); sys.exit(1)
a = base64.b64decode(conv["data"]); b = base64.b64decode(diff["data"])
if len(a) != len(b):
    print("FAIL: diffusion decode changed the byte count"); sys.exit(1)
if a == b:
    print("FAIL: decoder=diffusion produced the conv decoder's bytes"); sys.exit(1)

# STATIC is the failure mode, and it is not a range check: noise spans 0..255
# happily. The sampler contract (x0, one step, timesteps x1000) is MEASURED —
# no pack ships a VAE config — and every wrong arm decodes to static, so the
# guard is the mean absolute difference between horizontally adjacent pixels
# (tests/lora_noise.py's metric, inlined here without numpy). It is compared
# RELATIVELY: the absolute value is a property of the canvas and step count
# (13.9 on a 4-step 9-frame 384x256 render, 2.2 on a 25-frame 8-step one), so
# the bar is the CONV arm's own number on the same clip. Measured ratios:
# 1.04-1.09 healthy, 16x on the broken sampler.
def gradient(buf, w, h, frames):
    total = 0
    count = 0
    row = w * 3
    for f in range(0, frames, max(1, frames // 4)):
        base = f * h * row
        for y in range(0, h, 8):
            o = base + y * row
            for x in range(0, w - 1):
                p = o + x * 3
                q = p + 3
                l = (buf[p] + buf[p + 1] + buf[p + 2]) / 3.0
                r = (buf[q] + buf[q + 1] + buf[q + 2]) / 3.0
                total += abs(l - r)
                count += 1
    return total / max(1, count)

gc = gradient(a, conv["width"], conv["height"], conv["frames"])
gd = gradient(b, diff["width"], diff["height"], diff["frames"])
print(f"adjacent-pixel gradient: conv {gc:.2f}  diffusion {gd:.2f}  ratio {gd / max(gc, 1e-6):.2f}")
if gd > gc * 2.0 + 2.0:
    print("FAIL: decoder=diffusion decoded to STATIC — check the sampler contract "
          "(MLX_SERVE_DIFFVAE_OUTPUT/STEPS/TSCALE)"); sys.exit(1)
print(f"PASS: decoder=diffusion decoded {diff['frames']}f, real picture, differs from conv")
PYEOF
  fi
elif [ "$code" = "400" ]; then
  grep -q "vae_diffusion_decoder" "$DV_OUT" \
    && echo "PASS: decoder=diffusion 400s by name when the pack does not ship it" \
    || { echo "FAIL: 400 body does not name the missing file"; head -c 300 "$DV_OUT"; rc=1; }
else
  echo "FAIL: decoder=diffusion returned $code without the decoder file (want 400)"; rc=1
fi

# An unknown decoder name is a named 400 on every pack.
code=$(curl -s --max-time 60 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
  -d "$DV_REQ,\"decoder\":\"nope\"}" -o /tmp/test_video_gen_baddec.json -w "%{http_code}")
[ "$code" = "400" ] && echo "PASS: unknown 'decoder' value 400s" \
  || { echo "FAIL: unknown 'decoder' returned $code (want 400)"; rc=1; }

# Optional: mux to mp4 if ffmpeg is present (proves a playable clip, with sound
# when an audio track was decoded above into /tmp/test_video_gen.wav).
if [ $rc -eq 0 ] && command -v ffmpeg >/dev/null 2>&1; then
  python3 -c "import json,base64;d=json.load(open('$OUT'));open('/tmp/tvg.rgb','wb').write(base64.b64decode(d['data']));print(d['width'],d['height'],d['frames'])" >/tmp/tvg.dims
  read W H F < /tmp/tvg.dims
  if [ -f /tmp/test_video_gen.wav ]; then
    ffmpeg -y -f rawvideo -pix_fmt rgb24 -s "${W}x${H}" -r 24 -i /tmp/tvg.rgb -frames:v "$F" \
      -i /tmp/test_video_gen.wav -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest /tmp/test_video_gen.mp4 >/dev/null 2>&1 \
      && echo "PASS: muxed /tmp/test_video_gen.mp4 (with audio)"
  else
    ffmpeg -y -f rawvideo -pix_fmt rgb24 -s "${W}x${H}" -r 24 -i /tmp/tvg.rgb -frames:v "$F" \
      -c:v libx264 -pix_fmt yuv420p /tmp/test_video_gen.mp4 >/dev/null 2>&1 \
      && echo "PASS: muxed /tmp/test_video_gen.mp4 (video only)"
  fi
fi
exit $rc
