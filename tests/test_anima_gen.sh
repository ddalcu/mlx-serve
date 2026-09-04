#!/usr/bin/env bash
# Anima image gen on the ONE main server: headless boot -> load the Anima
# model by absolute path -> generate -> assert a valid PNG of the requested
# size -> server survives -> img2img generates too (Qwen-Image VAE encoder)
# -> 'guidance' (CFG scale) and 'negative_prompt' each change the output ->
# out-of-range 'guidance' 400s -> instruction edit is a clean error (no edit
# training) -> unload. Proves the image-backend seam routes `anima` to the
# Anima engine end to end (tokenizers, adapter, DiT, VAE encode+decode, PNG
# encode).
#
# Skips gracefully when no Anima model is present. The Anima dir must be a
# pack produced by scripts/convert_anima_weights.py (config.json with
# {"model_type":"anima",...} + transformer.safetensors + text_encoder.safetensors
# + tokenizer/ + vae.safetensors + t5_tokenizer/).
#
# Usage: ANIMA_MODEL=<dir> ./tests/test_anima_gen.sh [port]
set -uo pipefail
PORT="${1:-11398}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

ANIMA="${ANIMA_MODEL:-}"
[ -n "$ANIMA" ] || { echo "SKIP: no Anima model (set ANIMA_MODEL to a converted pack dir)"; exit 0; }
[ -f "$ANIMA/config.json" ] || { echo "SKIP: $ANIMA has no config.json (run scripts/convert_anima_weights.py)"; exit 0; }
[ -f "$ANIMA/transformer.safetensors" ] || { echo "SKIP: $ANIMA has no transformer.safetensors"; exit 0; }

# Headless: --model-dir anywhere; the empty HF hub discovers 0 models (load-by-path case).
HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_anima_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 /tmp/test_anima_server.log; exit 1; }
  sleep 1
done

api() { curl -s -m 1200 "http://127.0.0.1:$PORT$1" "${@:2}"; }
ANIMA_ID="$(basename "$ANIMA")"

# 1. Load Anima by absolute path -> ready with "image" capability.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$ANIMA\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$ANIMA_ID' and x['state']=='ready' and 'image' in x.get('capabilities',[])]
assert m, 'Anima not ready with image cap: '+json.dumps(d)
print('PASS: load-model by path -> Anima ready, capabilities', m[0]['capabilities'])
"

# 2. Generate at 512x512 -> PNG of that size.
api /v1/images/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ANIMA_ID\",\"prompt\":\"a red apple on a wooden table, simple illustration\",\"size\":\"512x512\",\"steps\":4,\"seed\":7}" \
  -o /tmp/test_anima_img.json -w ''
python3 - /tmp/test_anima_img.json <<'PY'
import sys,json,base64,struct
b=base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8]==bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
w,h=struct.unpack(">II", b[16:24])
assert (w,h)==(512,512), f"expected 512x512, got {w}x{h}"
print(f"PASS: /v1/images/generations (Anima) -> {len(b)} byte PNG {w}x{h}")
PY

# 3. Server survives the gen.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after Anima gen"; exit 1; }

# 4. img2img: VAE-encode a source image, renoise at strength, generate ->
# a valid PNG at the requested size (the Qwen-Image VAE encoder path).
SRC=/tmp/test_anima_i2i_src.png
python3 - "$SRC" <<'PY'
import sys, struct, zlib
W = H = 64
row = bytearray([0]) + bytes([128, 64, 32] * W)
rows = bytes(row) * H
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
open(sys.argv[1], "wb").write(
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
PY
python3 - "$SRC" /tmp/test_anima_i2i_req.json <<'PY'
import sys, json, base64
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"model": "ANIMA_ID_PLACEHOLDER", "prompt": "a red apple on a wooden table, simple illustration",
           "image": b64, "mode": "variation", "strength": 0.6, "size": "512x512", "steps": 4, "seed": 7},
          open(sys.argv[2], "w"))
PY
sed -i '' "s/ANIMA_ID_PLACEHOLDER/$ANIMA_ID/" /tmp/test_anima_i2i_req.json
CODE=$(api /v1/images/generations -X POST -H 'Content-Type: application/json' -d @/tmp/test_anima_i2i_req.json -o /tmp/test_anima_i2i_resp.json -w '%{http_code}')
[ "$CODE" = "200" ] || { echo "FAIL: img2img on Anima should generate, got HTTP $CODE"; cat /tmp/test_anima_i2i_resp.json; exit 1; }
python3 - /tmp/test_anima_i2i_resp.json <<'PY'
import sys,json,base64,struct
b=base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8]==bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
w,h=struct.unpack(">II", b[16:24])
assert (w,h)==(512,512), f"expected 512x512, got {w}x{h}"
print(f"PASS: img2img (Anima, VAE encoder) -> {len(b)} byte PNG {w}x{h}")
PY
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after img2img"; exit 1; }

# 5. `guidance` (CFG scale) and `negative_prompt` actually reach the DiT.
# Before this wiring existed, both fields were silently ignored (cfg always
# fell back to the pack's own recommended_cfg regardless of the body, and
# there was no unconditional-branch text to steer) — same seed/prompt would
# render BYTE-IDENTICAL images no matter what these fields said. The
# assertion is that they now render DIFFERENT images.
req() { # req <guidance> <negative_prompt|""> <out.json>
  python3 -c "
import json, sys
d = {'model': '$ANIMA_ID', 'prompt': 'a red apple on a wooden table, simple illustration',
     'size': '512x512', 'steps': 2, 'seed': 11, 'guidance': float(sys.argv[1])}
neg = sys.argv[2]
if neg:
    d['negative_prompt'] = neg
json.dump(d, open(sys.argv[3], 'w'))
" "$1" "$2" "$3"
}
png_bytes() { python3 -c "import sys,json,base64; print(len(base64.b64decode(json.load(open(sys.argv[1]))['data'][0]['b64_json'])))" "$1"; }
png_b64() { python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['data'][0]['b64_json'])" "$1"; }

req 1.0 "" /tmp/test_anima_cfg1.json
api /v1/images/generations -X POST -H 'Content-Type: application/json' -d @/tmp/test_anima_cfg1.json -o /tmp/test_anima_cfg1_resp.json -w ''
req 6.0 "" /tmp/test_anima_cfg6.json
api /v1/images/generations -X POST -H 'Content-Type: application/json' -d @/tmp/test_anima_cfg6.json -o /tmp/test_anima_cfg6_resp.json -w ''
[ "$(png_b64 /tmp/test_anima_cfg1_resp.json)" != "$(png_b64 /tmp/test_anima_cfg6_resp.json)" ] || { echo "FAIL: 'guidance' had no effect on Anima output (cfg 1.0 vs 6.0 identical)"; exit 1; }
echo "PASS: 'guidance' changes Anima output (cfg 1.0 vs 6.0 differ, $(png_bytes /tmp/test_anima_cfg1_resp.json) vs $(png_bytes /tmp/test_anima_cfg6_resp.json) bytes)"

req 6.0 "" /tmp/test_anima_neg0.json
api /v1/images/generations -X POST -H 'Content-Type: application/json' -d @/tmp/test_anima_neg0.json -o /tmp/test_anima_neg0_resp.json -w ''
req 6.0 "green, plant, leaf, foliage" /tmp/test_anima_neg1.json
api /v1/images/generations -X POST -H 'Content-Type: application/json' -d @/tmp/test_anima_neg1.json -o /tmp/test_anima_neg1_resp.json -w ''
[ "$(png_b64 /tmp/test_anima_neg0_resp.json)" != "$(png_b64 /tmp/test_anima_neg1_resp.json)" ] || { echo "FAIL: 'negative_prompt' had no effect on Anima output"; exit 1; }
echo "PASS: 'negative_prompt' changes Anima output at fixed cfg/seed"
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after guidance/negative_prompt gens"; exit 1; }

# 5b. Out-of-range 'guidance' is a named 400, not a crash or a silent clamp.
CODE=$(api /v1/images/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$ANIMA_ID\",\"prompt\":\"x\",\"size\":\"512x512\",\"steps\":2,\"guidance\":50}" \
  -o /tmp/test_anima_badcfg_resp.json -w '%{http_code}')
[ "$CODE" = "400" ] || { echo "FAIL: out-of-range 'guidance' should 400, got HTTP $CODE"; cat /tmp/test_anima_badcfg_resp.json; exit 1; }
echo "PASS: out-of-range 'guidance' -> HTTP 400"

# 6. Instruction editing (mode:"edit") is a clean error — Anima has no edit
# training, unlike img2img this stays a 400 by design.
python3 -c "
import json
d = json.load(open('/tmp/test_anima_i2i_req.json'))
d['mode'] = 'edit'
json.dump(d, open('/tmp/test_anima_edit_req.json', 'w'))
"
CODE=$(api /v1/images/generations -X POST -H 'Content-Type: application/json' -d @/tmp/test_anima_edit_req.json -o /tmp/test_anima_edit_resp.json -w '%{http_code}')
[ "$CODE" -ge 400 ] || { echo "FAIL: instruction edit on Anima should be a clean error, got HTTP $CODE"; cat /tmp/test_anima_edit_resp.json; exit 1; }
echo "PASS: instruction edit on Anima -> HTTP $CODE (no edit training, not a crash)"
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after the rejected edit"; exit 1; }

# 7. Unload.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$ANIMA_ID\"}" >/dev/null
echo "PASS: unload"

echo "ALL PASS: test_anima_gen.sh"
