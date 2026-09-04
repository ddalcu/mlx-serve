#!/usr/bin/env bash
# FLUX.1-dev / FLUX.1-schnell native text→image, end to end over HTTP.
#
# The FLUX.1 pipeline is its own image backend (T5-XXL + CLIP-L text encoders,
# per-block-modulated MMDiT, 16-channel VAE) — architecturally distinct from
# FLUX.2 klein. This pins that a FLUX.1 mflux pack (which ships NO root
# config.json) is discovered, loaded, advertises the image capability, and
# generates a valid, non-degenerate PNG at the requested size.
#
# Asserts:
#   1. the pack loads by absolute path and reports an image capability
#   2. /v1/images/generations returns a decodable PNG of the right dimensions
#   3. the image is not a constant field (a real render has spatial variance)
#   4. the FLUX.1 backend actually engaged (the [flux1] dit boot line) — a
#      silent fall-through to another image backend can't be caught by the PNG
#
# SKIPs without a FLUX.1 pack (default: the q4 dev mirror).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-11322}"
MODEL_DIR="${FLUX1_MODEL:-$HOME/.mlx-serve/models/mflux-community/flux-1-dev-mflux-q4}"
BIN="$ROOT/zig-out/bin/mlx-serve"
TMP="$(mktemp -d)"
SRV_PID=""
trap 'kill $SRV_PID 2>/dev/null || true; rm -rf "$TMP"' EXIT

pass() { echo "  ✅ $1"; }
fail() { echo "  ❌ $1"; exit 1; }

# The FLUX.1 pack is identified by its transformer weights, not a config.json.
[ -f "$MODEL_DIR/transformer/model.safetensors.index.json" ] || { echo "SKIP: no FLUX.1 pack at $MODEL_DIR"; exit 0; }
[ -x "$BIN" ] || fail "build first: zig build -Doptimize=ReleaseFast"
command -v python3 >/dev/null || fail "python3 required"

LOG="$TMP/server.log"
env "$BIN" --serve --port "$PORT" --model-dir "$HOME/.mlx-serve/models" --log-level info >"$LOG" 2>&1 &
SRV_PID=$!
for _ in $(seq 1 30); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done

echo "── load FLUX.1 pack ──"
LOAD_JSON="$(curl -fsS -X POST "http://127.0.0.1:$PORT/v1/load-model" -H 'Content-Type: application/json' \
  --max-time 300 -d "{\"model\":\"$MODEL_DIR\"}")" || fail "load-model failed (see $LOG)"
MODEL_ID="$(echo "$LOAD_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['model']['id'])")" || fail "no model id in load response"
# Capabilities are advertised on /v1/models (not the load response).
curl -fsS "http://127.0.0.1:$PORT/v1/models" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); m=[x for x in d['data'] if x['id']==sys.argv[1]][0]; sys.exit(0 if 'image' in m.get('capabilities',[]) else 1)" "$MODEL_ID" \
  || fail "loaded model does not advertise the image capability on /v1/models"
pass "loaded $MODEL_ID with image capability"

echo "── generate ──"
SIZE="${FLUX1_TEST_SIZE:-512}"
STEPS="${FLUX1_TEST_STEPS:-8}"
REQ="{\"model\":\"$MODEL_ID\",\"prompt\":\"a photo of a red fox in fresh snow at golden hour\",\"size\":\"${SIZE}x${SIZE}\",\"steps\":$STEPS,\"seed\":7}"
curl -fsS -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' \
  --max-time 590 -d "$REQ" \
  | python3 -c "import json,base64,sys; d=json.load(sys.stdin); open('$TMP/out.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))" \
  || fail "generation failed (see $LOG)"
pass "PNG generated"

python3 - "$TMP/out.png" "$SIZE" <<'PY' || fail "PNG invalid or degenerate"
import sys, struct, zlib
path, size = sys.argv[1], int(sys.argv[2])
data = open(path, "rb").read()
assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
# IHDR width/height
w, h = struct.unpack(">II", data[16:24])
assert (w, h) == (size, size), f"dims {w}x{h} != {size}x{size}"
# Decode enough to check the image is not a constant field: gather all IDAT,
# inflate, and measure the spread of raw filtered bytes (cheap proxy).
idat = b""
i = 8
while i < len(data):
    ln = struct.unpack(">I", data[i:i+4])[0]
    typ = data[i+4:i+8]
    if typ == b"IDAT":
        idat += data[i+8:i+8+ln]
    i += 12 + ln
raw = zlib.decompress(idat)
lo, hi = min(raw), max(raw)
assert hi - lo > 12, f"image looks constant (byte spread {hi-lo})"
print(f"  PNG {w}x{h}, byte spread {hi-lo}")
PY
pass "PNG is a valid, non-degenerate ${SIZE}x${SIZE} image"

grep -q "\[flux1\] dit:" "$LOG" || fail "FLUX.1 backend did not engage ([flux1] dit line absent)"
pass "FLUX.1 backend engaged"

# ── Runtime LoRA (stacked, summed at forward) ──
# Synthesize a rank-4 diffusers-named FLUX.1 adapter (both A and B non-zero so
# the delta actually moves the image), attach it, and assert it (1) matched
# real modules and (2) changed the output vs the same-seed baseline. This pins
# the whole path: lora.Arch.flux1 canonicalization + flux1.attachLora.
echo "── LoRA ──"
LORA="$TMP/flux1_lora.safetensors"
python3 - "$LORA" <<'PY'
import json, struct, math, sys
R, D = 4, 3072
def vals(n, seed): return [0.02*math.sin(0.1*(i+seed)) for i in range(n)]
mods = ["transformer.transformer_blocks.0.attn.to_q",
        "transformer.transformer_blocks.0.attn.to_v",
        "transformer.single_transformer_blocks.0.attn.to_q"]
tensors = {}
for j, m in enumerate(mods):
    tensors[m+".lora_A.weight"] = ([R, D], vals(R*D, j*7+1))
    tensors[m+".lora_B.weight"] = ([D, R], vals(D*R, j*7+100))
header, blob = {}, bytearray()
for name, (shape, data) in tensors.items():
    raw = struct.pack("<%df" % len(data), *data)
    header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(blob), len(blob)+len(raw)]}
    blob += raw
hb = json.dumps(header).encode()
open(sys.argv[1], "wb").write(struct.pack("<Q", len(hb)) + hb + bytes(blob))
PY
sha() { python3 -c "import json,base64,sys,hashlib; print(hashlib.sha256(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json'])).hexdigest())"; }
BASE_SHA="$(curl -fsS -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' --max-time 590 \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"a cat\",\"size\":\"256x256\",\"steps\":4,\"seed\":42}" | sha)" || fail "baseline gen failed"
LORA_SHA="$(curl -fsS -X POST "http://127.0.0.1:$PORT/v1/images/generations" -H 'Content-Type: application/json' --max-time 590 \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"a cat\",\"size\":\"256x256\",\"steps\":4,\"seed\":42,\"lora_paths\":[\"$LORA\"],\"lora_scales\":[1.0]}" | sha)" || fail "LoRA gen failed"
grep -q "\[image\] lora: matched" "$LOG" || fail "LoRA did not attach (no '[image] lora: matched' line)"
[ "$BASE_SHA" != "$LORA_SHA" ] || fail "LoRA changed nothing — same output as baseline"
pass "LoRA attached and changed the output (${BASE_SHA:0:12} != ${LORA_SHA:0:12})"

echo
echo "✅ FLUX.1 image-gen test passed."
