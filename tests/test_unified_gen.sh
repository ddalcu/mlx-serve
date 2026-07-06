#!/usr/bin/env bash
# Unified media-gen on the ONE main server: headless boot -> load a gen model by
# absolute path -> generate -> unload -> assert unloaded, all on one process/port.
# Also asserts a chat model and a gen model can be RESIDENT TOGETHER and that an
# LLM chat stream still works while a gen model is loaded (decisions: headless
# auto-start, load->gen->unload, coexist-with-chat).
#
# Skips gracefully when no media model is present. Prefers FLUX (image) for the
# generate assertion because its output is deterministic-shaped (PNG magic+dims)
# and fast; video/audio have their own scripts (test_video_gen.sh / test_tts.sh).
#
# Usage: FLUX_MODEL=<dir> CHAT_MODEL=<dir> ./tests/test_unified_gen.sh [port]
set -uo pipefail
PORT="${1:-11398}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

FLUX="${FLUX_MODEL:-$(ls -d ~/.cache/huggingface/hub/models--Runpod--FLUX.2-klein-4B-mflux-4bit/snapshots/* 2>/dev/null | head -1)}"
CHAT="${CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
[ -n "$FLUX" ] || { echo "SKIP: no FLUX model (set FLUX_MODEL)"; exit 0; }

# Headless: --model-dir points anywhere (the HF hub is fine — its nested layout
# discovers 0 models, which is exactly the "load purely by path" case).
HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_unified_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 /tmp/test_unified_server.log; exit 1; }
  sleep 1
done

api() { curl -s -m 600 "http://127.0.0.1:$PORT$1" "${@:2}"; }

# 1. Headless: no default model -> empty model list.
N=$(api /v1/models | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["data"]))')
[ "$N" = "0" ] || { echo "FAIL: headless /v1/models should be empty, got $N"; exit 1; }
echo "PASS: headless boot, /v1/models empty (no primary model)"

FLUX_ID="$(basename "$FLUX")"

# 2. Load FLUX by absolute path -> ready with "image" capability.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$FLUX\"}" >/dev/null
api /v1/models | python3 -c '
import sys,json
d=json.load(sys.stdin)["data"]
m=[x for x in d if x["state"]=="ready" and "image" in x.get("capabilities",[])]
assert m, "FLUX not ready with image cap: "+json.dumps(d)
print("PASS: load-model by path -> FLUX ready, capabilities", m[0]["capabilities"])
'

# 3. Generate an image -> valid PNG.
api /v1/images/generations -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$FLUX_ID\",\"prompt\":\"a green triangle\",\"steps\":4}" -o /tmp/test_unified_img.json -w ''
python3 - /tmp/test_unified_img.json <<'PY'
import sys,json,base64,struct
b=base64.b64decode(json.load(open(sys.argv[1]))["data"][0]["b64_json"])
assert b[:8]==bytes([0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A]), "not a PNG"
w,h=struct.unpack(">II", b[16:24])
print(f"PASS: /v1/images/generations on the unified server -> {len(b)} byte PNG {w}x{h}")
PY

# 4. Server survives a gen, still healthy.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after image gen"; exit 1; }

# 4b. A CHAT request aimed at the resident image model must 400 with an
#     actionable message — pre-guard this SIGSEGV'd the whole server (the gen
#     stub tokenizer yields 0 tokens and prefill derefs a null transformer):
#     one remote request could kill the process. Same for /v1/messages and
#     the Ollama surface, which share the textGenRejectReason guard.
CHAT_AT_FLUX=$(api /v1/chat/completions -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$FLUX_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}")
echo "$CHAT_AT_FLUX" | grep -q "images/generations" || { echo "FAIL: chat at image model should 400 with a redirect, got: $(echo "$CHAT_AT_FLUX" | head -c 200)"; exit 1; }
MSG_AT_FLUX=$(api /v1/messages -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$FLUX_ID\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
echo "$MSG_AT_FLUX" | grep -q "invalid_request_error" || { echo "FAIL: /v1/messages at image model should 400, got: $(echo "$MSG_AT_FLUX" | head -c 200)"; exit 1; }
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server DIED on chat-at-image-model (the SIGSEGV class is back)"; exit 1; }
echo "PASS: chat/messages at the image model -> clean 400s, server alive"

# 5. Coexistence: load a chat model alongside FLUX, BOTH resident, chat streams.
if [ -n "$CHAT" ]; then
  CHAT_ID="$(basename "$CHAT")"
  api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$CHAT\"}" >/dev/null
  api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
ready={x['id'] for x in d if x['state']=='ready'}
assert '$FLUX_ID' in ready and '$CHAT_ID' in ready, 'both should be resident: '+json.dumps([(x['id'],x['state']) for x in d])
print('PASS: chat + gen model RESIDENT together:', sorted(ready))
"
  TOK=$(curl -s -m 120 -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$CHAT_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16,\"stream\":true}" \
    | grep -c '"content":')
  [ "$TOK" -ge 1 ] || { echo "FAIL: chat did not stream while gen model resident"; exit 1; }
  echo "PASS: chat streams ($TOK content deltas) with a gen model also resident"
fi

# 6. Unload FLUX -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$FLUX_ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$FLUX_ID']
assert m and m[0]['state']=='unloaded', 'FLUX should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> FLUX unloaded (stub retained)')
"

echo "ALL PASS: unified media-gen (headless boot, load->gen->unload, coexistence)"
