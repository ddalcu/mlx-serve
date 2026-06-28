#!/usr/bin/env bash
# Krea-2-Turbo image gen on the ONE main server: headless boot -> load the Krea
# model by absolute path -> generate (at a NON-1024 size, exercising the relaxed
# Krea size gate) -> assert a valid PNG of the requested size -> coexist with a
# chat model -> unload. Proves the image-backend seam routes `krea*` to the Krea
# engine and that chat + Krea coexist on one process/port.
#
# Skips gracefully when no Krea model is present. The Krea dir must be assembled
# (transformer_*.safetensors + text_encoder/ + vae/ + tokenizer/ + config.json
# with {"model_type":"krea2_turbo"}).
#
# Usage: KREA_MODEL=<dir> CHAT_MODEL=<dir> ./tests/test_krea_gen.sh [port]
set -uo pipefail
PORT="${1:-11399}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

KREA="${KREA_MODEL:-$(ls -d ~/.mlx-serve/models/avlp12/Krea-2-Turbo-Alis-MLX-mixed-4-8 2>/dev/null | head -1)}"
CHAT="${CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
[ -n "$KREA" ] || { echo "SKIP: no Krea model (set KREA_MODEL to an assembled dir)"; exit 0; }
[ -f "$KREA/config.json" ] || { echo "SKIP: $KREA has no config.json (assemble the dir + add {\"model_type\":\"krea2_turbo\"})"; exit 0; }

# Headless: --model-dir anywhere; the empty HF hub discovers 0 models (load-by-path case).
HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_krea_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 /tmp/test_krea_server.log; exit 1; }
  sleep 1
done

api() { curl -s -m 1200 "http://127.0.0.1:$PORT$1" "${@:2}"; }
KREA_ID="$(basename "$KREA")"

# 1. Headless: no default model.
N=$(api /v1/models | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["data"]))')
[ "$N" = "0" ] || { echo "FAIL: headless /v1/models should be empty, got $N"; exit 1; }
echo "PASS: headless boot, /v1/models empty"

# 2. Load Krea by absolute path -> ready with "image" capability.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$KREA\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$KREA_ID' and x['state']=='ready' and 'image' in x.get('capabilities',[])]
assert m, 'Krea not ready with image cap: '+json.dumps(d)
print('PASS: load-model by path -> Krea ready, capabilities', m[0]['capabilities'])
"

# 3. Generate at 512x512 (NON-1024 -> exercises the relaxed Krea size gate) -> PNG of that size.
api /v1/images/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$KREA_ID\",\"prompt\":\"a red fox in the snow\",\"size\":\"512x512\",\"steps\":8}" \
  -o /tmp/test_krea_img.json -w ''
python3 - /tmp/test_krea_img.json <<'PY'
import sys,json,base64,struct
b=base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8]==bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
w,h=struct.unpack(">II", b[16:24])
assert (w,h)==(512,512), f"expected 512x512, got {w}x{h}"
print(f"PASS: /v1/images/generations (Krea, relaxed size) -> {len(b)} byte PNG {w}x{h}")
PY

# 4. Server survives the gen.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after Krea gen"; exit 1; }

# 5. Coexistence with a chat model.
if [ -n "$CHAT" ]; then
  CHAT_ID="$(basename "$CHAT")"
  api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$CHAT\"}" >/dev/null
  api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
ready={x['id'] for x in d if x['state']=='ready'}
assert '$KREA_ID' in ready and '$CHAT_ID' in ready, 'both should be resident: '+json.dumps([(x['id'],x['state']) for x in d])
print('PASS: chat + Krea RESIDENT together:', sorted(ready))
"
  TOK=$(curl -s -m 120 -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$CHAT_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16,\"stream\":true}" \
    | grep -c '"content":')
  [ "$TOK" -ge 1 ] || { echo "FAIL: chat did not stream while Krea resident"; exit 1; }
  echo "PASS: chat streams ($TOK content deltas) with Krea also resident"
fi

# 6. Unload Krea -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$KREA_ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$KREA_ID']
assert m and m[0]['state']=='unloaded', 'Krea should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> Krea unloaded (stub retained)')
"

echo "ALL PASS: Krea-2-Turbo image gen (headless boot, load->gen->unload, relaxed size, coexistence)"
