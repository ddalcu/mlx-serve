#!/usr/bin/env bash
# Native FLUX.2 image endpoint smoke test: text-to-image (JSON + SSE),
# image-to-image (structure retention at low strength + engagement log),
# conditioning rebalance (cond_gain/cond_weights + count validation), and
# runtime LoRA (synthetic zero-B adapter attaches; non-matching adapter 400s).
# Usage: FLUX_MODEL=<dir> ./tests/test_image_gen.sh [port]
set -uo pipefail
PORT="${1:-11399}"
MODEL="${FLUX_MODEL:-$(ls -d ~/.cache/huggingface/hub/models--Runpod--FLUX.2-klein-4B-mflux-4bit/snapshots/* 2>/dev/null | head -1)}"
[ -n "$MODEL" ] || { echo "SKIP: no FLUX model (set FLUX_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

"$BIN" --model "$MODEL" --serve --port "$PORT" >/tmp/test_image_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
# wait for /health (model load is heavy)
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 /tmp/test_image_server.log; exit 1; }
  sleep 2
done

OUT=/tmp/test_image_gen.json
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple on a wooden table","size":"1024x1024"}' -o "$OUT" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: http $code"; head -c 300 "$OUT"; exit 1; }
# decode b64 PNG, check magic + dims
python3 - "$OUT" <<'PY'
import sys, json, base64, struct
d=json.load(open(sys.argv[1]))
b=base64.b64decode(d["data"][0]["b64_json"])
assert b[:8]==bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
w,h=struct.unpack(">II", b[16:24])
print(f"PASS: /v1/images/generations -> {len(b)} byte PNG {w}x{h}")
assert w==1024 and h==1024, f"bad dims {w}x{h}"
PY

# SSE streaming: per-step progress events, then a complete event with the PNG.
SSE=/tmp/test_image_sse.txt
curl -sN -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple","size":"1024x1024","steps":4,"stream":true}' >"$SSE"
python3 - "$SSE" <<'PY'
import sys, json, base64
prog = 0; complete = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data: "): continue
    ev = json.loads(line[6:])
    if ev["type"] == "progress": prog += 1; assert {"stage","step","total"} <= set(ev)
    elif ev["type"] == "complete": complete = ev
assert prog >= 4, f"expected several progress events, got {prog}"
assert complete is not None, "no complete event"
png = base64.b64decode(complete["data"][0]["b64_json"])
assert png[:8] == bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "complete PNG bad"
print(f"PASS: SSE stream -> {prog} progress events + complete PNG ({len(png)} bytes)")
PY
[ $? -eq 0 ] || exit 1

# ── image-to-image: a high-contrast half-dark/half-bright source at LOW
# strength must retain its brightness split (the LTX I2V live-check pattern).
SRC=/tmp/test_img2img_src.png
python3 - "$SRC" <<'PY'
import sys, struct, zlib
W = H = 1024
rows = b""
for y in range(H):
    row = bytearray([0])
    for x in range(W):
        v = 235 if x >= W // 2 else 20
        row += bytes([v, v, v])
    rows += bytes(row)
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows))
       + chunk(b"IEND", b""))
open(sys.argv[1], "wb").write(png)
PY
REQ=/tmp/test_img2img_req.json
python3 - "$SRC" "$REQ" <<'PY'
import sys, json, base64
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"prompt": "a photo", "size": "1024x1024", "steps": 4,
           "strength": 0.2, "image": b64, "seed": 7}, open(sys.argv[2], "w"))
PY
OUT2=/tmp/test_img2img_out.json
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d @"$REQ" -o "$OUT2" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: img2img http $code"; head -c 300 "$OUT2"; exit 1; }
grep -q "\[image\] img2img:" /tmp/test_image_server.log || { echo "FAIL: no img2img engagement log line"; exit 1; }
# Shared checker: the output must keep the source's dark-left/bright-right split.
cat > /tmp/check_split.py <<'PY'
import sys, json, base64, zlib, struct
label = sys.argv[2]
d = json.load(open(sys.argv[1]))
png = base64.b64decode(d["data"][0]["b64_json"])
assert png[:8] == bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
pos, idat, w, h = 8, b"", 0, 0
while pos < len(png):
    ln, typ = struct.unpack(">I4s", png[pos:pos+8]); data = png[pos+8:pos+8+ln]; pos += 12 + ln
    if typ == b"IHDR": w, h, _, ct = struct.unpack(">IIBB", data[:10]); assert ct == 2
    elif typ == b"IDAT": idat += data
raw = zlib.decompress(idat)
stride = w * 3
prev = bytearray(stride)
left_sum = right_sum = 0; n = 0
for y in range(h):
    f = raw[y * (stride + 1)]
    line = bytearray(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)])
    for i in range(stride):
        a = line[i - 3] if i >= 3 else 0
        b = prev[i]
        c = prev[i - 3] if i >= 3 else 0
        if f == 1: line[i] = (line[i] + a) & 0xFF
        elif f == 2: line[i] = (line[i] + b) & 0xFF
        elif f == 3: line[i] = (line[i] + (a + b) // 2) & 0xFF
        elif f == 4:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            line[i] = (line[i] + pr) & 0xFF
    prev = line
    if y % 32 == 0:
        for x in range(0, w // 2, 16): left_sum += line[x * 3]; n += 1
        for x in range(w // 2, w, 16): right_sum += line[x * 3]
left, right = left_sum / n, right_sum / n
print(f"PASS: {label} kept the split (left mean {left:.0f}, right mean {right:.0f})")
assert right - left > 60, f"{label} lost the source structure (left {left:.0f}, right {right:.0f})"
PY
python3 /tmp/check_split.py "$OUT2" "img2img strength=0.2" || exit 1

# ── strength out of range → 400
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"x","image":"aGk=","strength":1.5}' -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: bad strength returned $code (want 400)"; exit 1; }
echo "PASS: strength 1.5 -> 400"

# ── conditioning rebalance: wrong weight count → 400 (FLUX taps 3 layers)
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple","steps":2,"cond_weights":"1 1 1 1"}' -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: wrong cond_weights count returned $code (want 400)"; exit 1; }
echo "PASS: 4 cond_weights on FLUX -> 400"

# ── baseline for the effect checks below: plain steps-2 seed-7 generation
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple","size":"1024x1024","steps":2,"seed":7}' \
  -o /tmp/test_plain2.json -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: baseline http $code"; exit 1; }

# ── valid rebalance engages AND changes the output (an engagement log alone
# can't see a silently-ignored parameter)
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple","size":"1024x1024","steps":2,"cond_gain":1.5,"cond_weights":[1,0.8,1.2],"seed":7}' \
  -o /tmp/test_rebalance.json -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: rebalance http $code"; head -c 300 /tmp/test_rebalance.json; exit 1; }
grep -q "\[image\] rebalance: gain=1.50" /tmp/test_image_server.log || { echo "FAIL: no rebalance engagement log"; exit 1; }
cmp -s /tmp/test_plain2.json /tmp/test_rebalance.json && { echo "FAIL: rebalance did not change the output"; exit 1; }
echo "PASS: cond_gain + 3 cond_weights -> 200, engagement log, output differs from baseline"

# ── runtime LoRA: synthetic zero-B adapter for one flux module attaches (200 +
# matched log); an adapter with foreign keys is a 400.
LORA_OK=/tmp/test_lora_ok.safetensors
LORA_BAD=/tmp/test_lora_bad.safetensors
python3 - "$LORA_OK" "$LORA_BAD" <<'PY'
import sys, json, struct
def write_st(path, tensors):
    header = {}
    blob = b""
    for name, (shape, data) in tensors.items():
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    hj = json.dumps(header).encode()
    open(path, "wb").write(struct.pack("<Q", len(hj)) + hj + blob)
r, out_dim, in_dim = 2, 3072, 3072
a = struct.pack(f"<{r*in_dim}f", *([0.01] * (r * in_dim)))
b = struct.pack(f"<{out_dim*r}f", *([0.0] * (out_dim * r)))   # zero-B: attach is a no-op numerically
write_st(sys.argv[1], {
    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": ([r, in_dim], a),
    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": ([out_dim, r], b),
})
write_st(sys.argv[2], {
    "unet.some_other_model.lora_A.weight": ([r, 8], struct.pack("<16f", *([0.0]*16))),
    "unet.some_other_model.lora_B.weight": ([8, r], struct.pack("<16f", *([0.0]*16))),
})
PY
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"a red apple\",\"size\":\"1024x1024\",\"steps\":2,\"lora_path\":\"$LORA_OK\",\"seed\":7}" \
  -o /tmp/test_lora.json -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: lora http $code"; head -c 300 /tmp/test_lora.json; exit 1; }
grep -q "\[image\] lora: matched 1 modules" /tmp/test_image_server.log || { echo "FAIL: no lora matched log"; exit 1; }
# zero-B delta is exactly 0 → output must be byte-identical to the baseline
# (proves the runtime-LoRA path is numerically transparent and the base
# weights weren't corrupted by attach)
cmp -s /tmp/test_plain2.json /tmp/test_lora.json || { echo "FAIL: zero-B LoRA changed the output"; exit 1; }
echo "PASS: zero-B LoRA attached (matched 1 module) and is byte-transparent"
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"x\",\"steps\":2,\"lora_path\":\"$LORA_BAD\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: foreign lora returned $code (want 400)"; exit 1; }
echo "PASS: non-matching LoRA -> 400"

# ── instruction edit (FLUX.2 in-context reference conditioning): generation
# starts from PURE NOISE, so the split can only come from the model attending
# to the clean reference tokens (probed live: diff ≈ 198).
EREQ=/tmp/test_edit_req.json
python3 - "$SRC" "$EREQ" <<'PY'
import sys, json, base64
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
json.dump({"prompt": "the same image, unchanged", "size": "1024x1024", "steps": 4,
           "mode": "edit", "image": b64, "seed": 7}, open(sys.argv[2], "w"))
PY
EOUT=/tmp/test_edit_out.json
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d @"$EREQ" -o "$EOUT" -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: edit http $code"; head -c 300 "$EOUT"; exit 1; }
grep -q "\[image\] edit:" /tmp/test_image_server.log || { echo "FAIL: no edit engagement log line"; exit 1; }
python3 /tmp/check_split.py "$EOUT" "instruction edit (pure-noise start)" || exit 1

# ── edit mode without an image → 400
code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"prompt":"x","mode":"edit"}' -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: imageless edit returned $code (want 400)"; exit 1; }
echo "PASS: mode=edit without image -> 400"
