#!/usr/bin/env bash
# Release battery: an agent keeps working while every media modality runs on the
# same server. The 2026-10-03 incident (a pi session "hung" for 10+ minutes while
# its own asset script ran a textured Hunyuan3D job) plus the rest of the media
# set, against a real agent-sized chat model with its DFlash drafter.
#
# One server, scratch HOME, `--log-level debug`:
#   chat  Qwen3.8-27B 4-bit + its shipped DFlash2 drafter, TWO concurrent agents
#         (pi-shaped turns: streaming, tools, low reasoning effort; one of them
#         alternating with plain turns), so batched decode runs beside media
#   media FLUX klein image, Qwen3-TTS speech, ACE-Step music, Hunyuan3D textured
#         res-320 mesh (optional, the incident's job)
# Asserts:
#   [1] every model loads, the drafter binds (`DFlash drafter ready`)
#   [2] solo references: one agent turn (dflash must engage), one seeded image
#   [3] the mix: all media jobs queued at once while both agents run turns back to back.
#       Every media request answers 200 with a valid payload, the image is
#       byte-identical to the solo one, at least MIX_MIN_CHATS turns finish
#       while media is running, every turn is clean (stream ends, tool args are
#       JSON, no markup in content), both agents land turns, the slowest first
#       token stays under MIX_MAX_TTFT, and the log shows `[gen-yield] engaged`
#   [4] cancel: a music client and a speech client hanging up mid-generation
#       each free the server
#   [5] alive: a final turn answers, no MLX error in the log
#
# Needs every pack below on a model root and the set inside this box's GPU
# budget; SKIPs otherwise. Overrides: MIX_CHAT_MODEL, MIX_IMAGE_MODEL,
# MIX_TTS_MODEL, MIX_MUSIC_MODEL, MIX_MESH_MODEL ("" skips the mesh job),
# MIX_MAX_TTFT (seconds, default 30), MIX_MIN_CHATS (default 3).
# Usage: ./tests/test_gen_chat_mix.sh [port]
set -uo pipefail
PORT="${1:-11463}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
source "$ROOT/tests/_lib_models.sh"

CHAT="${MIX_CHAT_MODEL:-$(find_model ddalcu/Qwen3.8-27B-MLX-Serve-4bit)}"
IMAGE="${MIX_IMAGE_MODEL:-$(find_model Runpod/FLUX.2-klein-4B-mflux-4bit)}"
TTS="${MIX_TTS_MODEL:-$(find_model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit)}"
MUSIC="${MIX_MUSIC_MODEL:-$(find_model ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit)}"
MESH="${MIX_MESH_MODEL-$(find_model ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit)}"
MAX_TTFT="${MIX_MAX_TTFT:-30}"
MIN_CHATS="${MIX_MIN_CHATS:-3}"
for m in "$CHAT" "$IMAGE" "$TTS" "$MUSIC"; do
  [ -n "$m" ] && [ -d "$m" ] || { echo "SKIP: a pack is missing (chat=$CHAT image=$IMAGE tts=$TTS music=$MUSIC)"; exit 0; }
done
[ -f "$CHAT/drafter/config.json" ] || { echo "SKIP: $CHAT ships no drafter/"; exit 0; }
if [ -n "$MESH" ] && [ ! -f "$MESH/paint/config.json" ]; then MESH=""; fi
MODELS=("$CHAT" "$IMAGE" "$TTS" "$MUSIC")
[ -z "$MESH" ] || MODELS+=("$MESH")
total=0
for m in "${MODELS[@]}"; do total=$((total + $(model_gb "$m"))); done
[ "$total" -le "$(max_model_gb)" ] || { echo "SKIP: the set is ${total} GB, this box admits $(max_model_gb) GB"; exit 0; }

TMP=$(mktemp -d)
LOG="$TMP/server.log"
mkdir -p "$TMP/home"
# The default cap of 3 resident models would evict the set into churn mid-mix.
HOME="$TMP/home" "$BIN" --serve --port "$PORT" --log-level debug --max-resident-models "${#MODELS[@]}" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $(jobs -p) 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
  sleep 1
done
URL="http://127.0.0.1:$PORT"
api() { curl -s -m 1800 "$URL$1" "${@:2}"; }
now() { python3 -c 'import time; print(time.time())'; }
fail() { echo "FAIL: $*"; echo "   log: $LOG"; exit 1; }

# One chat turn; prints a JSON line {ok, ttft, total, kind, tool_calls, err}.
cat >"$TMP/turn.py" <<'PY'
import json, sys, time, urllib.request
url, model, kind = sys.argv[1], sys.argv[2], sys.argv[3]
task = sys.argv[4] if len(sys.argv) > 4 else "Open src/game.js and tell me what the main loop does."
tools = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
if kind == "agent":
    body = {"model": model, "stream": True, "max_tokens": 512, "reasoning_effort": "low", "tools": tools,
            "messages": [{"role": "system", "content": "You are a coding agent. Use tools to inspect files before answering."},
                         {"role": "user", "content": task}]}
else:
    body = {"model": model, "stream": True, "max_tokens": 64, "temperature": 0, "enable_thinking": False,
            "messages": [{"role": "user", "content": "Name three primary colors, comma separated."}]}
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time(); ttft = None; content = ""; reasoning = ""; calls = {}; finish = None; done = False; err = None
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "): continue
            if line == "data: [DONE]": done = True; break
            ch = json.loads(line[6:])["choices"]
            if not ch: continue
            d = ch[0].get("delta", {})
            if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                ttft = time.time() - t0
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                f = tc.get("function", {})
                c["name"] += f.get("name") or ""; c["args"] += f.get("arguments") or ""
            finish = ch[0].get("finish_reason") or finish
except Exception as e:
    err = repr(e)
if err is None:
    if not done or finish is None: err = f"stream ended without [DONE]/finish_reason (finish={finish})"
    elif any(m in content for m in ("<tool_call>", "<function=", "<think>", "</think>")): err = "markup in content: " + content[:200]
    elif not content.strip() and not calls and not (finish == "length" and reasoning): err = "empty reply"
    else:
        for c in calls.values():
            try: json.loads(c["args"] or "{}")
            except ValueError: err = "tool args not JSON: " + c["args"][:200]
print(json.dumps({"ok": err is None, "ttft": ttft, "total": time.time() - t0, "kind": kind, "agent": sys.argv[5] if len(sys.argv) > 5 else "",
                  "tool_calls": len(calls), "err": err}))
PY

# Validates one media response file by kind; prints a short description.
cat >"$TMP/media.py" <<'PY'
import base64, json, struct, sys
kind, path = sys.argv[1], sys.argv[2]
raw = open(path, "rb").read()
if kind == "image":
    png = base64.b64decode(json.loads(raw)["data"][0]["b64_json"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    print(f"PNG {struct.unpack('>II', png[16:24])}")
elif kind in ("speech", "music"):
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "not a WAV: " + raw[:120].decode("utf-8", "replace")
    rate, = struct.unpack("<I", raw[24:28]); ch, = struct.unpack("<H", raw[22:24])
    secs = (len(raw) - 44) / (rate * ch * 2)
    assert secs > 1.0, f"{secs:.2f}s of audio"
    print(f"WAV {secs:.1f}s {rate} Hz {ch}ch")
elif kind == "mesh":
    glb = base64.b64decode(json.loads(raw)["data"])
    assert glb[:4] == b"glTF", "not a GLB"
    doc = json.loads(glb[20:20 + struct.unpack("<I", glb[12:16])[0]])
    prim = doc["meshes"][0]["primitives"][0]
    assert "TEXCOORD_0" in prim["attributes"], "untextured GLB"
    print(f"GLB {doc['accessors'][prim['indices']]['count'] // 3} faces, textured")
PY

# [1] Load everything; the chat model brings its shipped drafter.
load() {
  local code
  code=$(api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$1\"}" -o "$TMP/load.json" -w "%{http_code}")
  [ "$code" = "200" ] || fail "load $1 http $code: $(head -c 300 "$TMP/load.json")"
}
for m in "${MODELS[@]}"; do load "$m"; done
grep -q "DFlash drafter ready" "$LOG" || fail "[1] the chat model's drafter did not bind"
! grep -q "\[registry\] evicting" "$LOG" || fail "[1] a load evicted another model: $(grep -m1 "\[registry\] evicting" "$LOG")"
echo "PASS: [1] chat (+DFlash), image, speech, music${MESH:+, mesh} loaded (${total} GB on disk)"

# [2] Solo references.
python3 "$TMP/turn.py" "$URL" "$CHAT" agent >"$TMP/solo_agent.json"
python3 -c "import json,sys; d=json.load(open('$TMP/solo_agent.json')); sys.exit(0 if d['ok'] else 1)" ||
  fail "[2] solo agent turn: $(cat "$TMP/solo_agent.json")"
grep -q "\[spec-stats\] mode=dflash" "$LOG" || fail "[2] the solo agent turn ran no dflash rounds"
IMG_REQ="{\"model\":\"$IMAGE\",\"prompt\":\"a pixel-art spaceship sprite on a black background\",\"size\":\"1024x1024\",\"steps\":16,\"seed\":7}"
api /v1/images/generations -H 'Content-Type: application/json' -d "$IMG_REQ" -o "$TMP/solo_image.json"
python3 "$TMP/media.py" image "$TMP/solo_image.json" >/dev/null || fail "[2] solo image"
echo "PASS: [2] solo agent turn $(python3 -c "import json; d=json.load(open('$TMP/solo_agent.json')); print(f\"{d['total']:.1f}s, ttft {d['ttft']:.2f}s\")"), solo image"

# [3] The mix. Media jobs go in at once (they run one after another on the
# inference thread); agent and plain turns alternate until the last one ends.
MARK=$(wc -c <"$LOG")
cat >"$TMP/speech.json" <<EOF
{"model":"$TTS","input":"Welcome back, pilot. The asteroid field ahead is dense, so keep your shields charged and your thrusters warm. Collect the blue crystals to upgrade your laser, avoid the red mines, and watch the radar for the mothership. When the warning siren sounds, you have ten seconds to reach the jump gate. Good luck out there, and remember: every star you pass is a story someone will tell."}
EOF
cat >"$TMP/music.json" <<EOF
{"model":"$MUSIC","prompt":"energetic chiptune space shooter theme, driving bass, arpeggiated leads","duration_seconds":30,"seed":7}
EOF
media() { # kind endpoint body-file
  ( code=$(api "$2" -H 'Content-Type: application/json' -d @"$3" -o "$TMP/mix_$1.out" -w "%{http_code}")
    echo "$code $(now)" >"$TMP/mix_$1.code" ) &
  MEDIA_PIDS+=($!)
}
MEDIA_PIDS=()
JOBS="image speech music"
if [ -n "$MESH" ]; then
  python3 - "$TMP/mesh.json" "$MESH" <<'PY'
import base64, json, struct, sys, zlib
W = H = 384
rows = b"".join(b"\x00" + b"".join(bytes([60, 60, 200]) if abs(x - 192) + abs(y - 192) < 140 else b"\xff\xff\xff" for x in range(W)) for y in range(H))
ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
png = b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(rows)) + ch(b"IEND", b"")
json.dump({"model": sys.argv[2], "image": base64.b64encode(png).decode(), "steps": 30, "octree_resolution": 320,
           "seed": 7, "texture": True}, open(sys.argv[1], "w"))
PY
  media mesh /v1/3d/generations "$TMP/mesh.json"
  JOBS="mesh $JOBS"
  sleep 0.5 # the incident's job goes first
fi
echo "$IMG_REQ" >"$TMP/image.json"
media image /v1/images/generations "$TMP/image.json"
media speech /v1/audio/speech "$TMP/speech.json"
media music /v1/audio/music-generations "$TMP/music.json"
MIX_START=$(now)

pending() { for j in $JOBS; do [ -f "$TMP/mix_$j.code" ] || return 0; done; return 1; }
agent() { # name alternate-plain task
  local n=0 kind
  while pending; do
    kind=agent
    [ "$2" = 1 ] && [ $((n % 2)) -eq 1 ] && kind=plain
    python3 "$TMP/turn.py" "$URL" "$CHAT" "$kind" "$3" "$1" >"$TMP/turn_$1.json"
    pending || break # finished after the last media job: not a mid-mix sample
    cat "$TMP/turn_$1.json" >>"$TMP/turns_$1.jsonl"; n=$((n + 1))
  done
}
: >"$TMP/turns_a.jsonl"; : >"$TMP/turns_b.jsonl"
agent a 1 "Open src/game.js and tell me what the main loop does." &
AG_A=$!
agent b 0 "Read package.json and list the build scripts it defines." &
AG_B=$!
wait "${MEDIA_PIDS[@]}" $AG_A $AG_B
cat "$TMP/turns_a.jsonl" "$TMP/turns_b.jsonl" >"$TMP/turns.jsonl"
MIX_SECS=$(python3 -c "print(round($(now) - $MIX_START))")

for j in $JOBS; do
  read -r code _ <"$TMP/mix_$j.code"
  [ "$code" = "200" ] || fail "[3] $j http $code: $(head -c 300 "$TMP/mix_$j.out")"
  desc=$(python3 "$TMP/media.py" "$j" "$TMP/mix_$j.out" 2>&1) || fail "[3] $j payload: $desc"
  echo "   $j: $desc"
done
python3 -c "
import json
a = json.load(open('$TMP/solo_image.json'))['data'][0]['b64_json']
b = json.load(open('$TMP/mix_image.out'))['data'][0]['b64_json']
raise SystemExit(0 if a == b else 1)" || fail "[3] the image changed under chat interleaving"
python3 - "$TMP/turns.jsonl" "$MAX_TTFT" "$MIN_CHATS" <<'PY' || fail "[3] chat during the mix (turns: $TMP/turns.jsonl)"
import json, sys
turns = [json.loads(l) for l in open(sys.argv[1])]
bad = [t for t in turns if not t["ok"]]
for t in bad: print("   bad turn:", t)
ttfts = [t["ttft"] for t in turns if t["ttft"] is not None]
slow = max(ttfts, default=0.0)
print(f"   {len(turns)} turns during the mix (agent a {sum(t['agent'] == 'a' for t in turns)}, agent b {sum(t['agent'] == 'b' for t in turns)}; "
      f"{sum(t['kind'] == 'agent' for t in turns)} agent-shaped, "
      f"{sum(t['tool_calls'] > 0 for t in turns)} with tool calls), slowest first token {slow:.1f}s, "
      f"slowest turn {max((t['total'] for t in turns), default=0):.1f}s")
assert not bad, "a turn failed"
assert len(turns) >= int(sys.argv[3]), f"only {len(turns)} turns landed while media ran"
assert all(any(t["agent"] == a for t in turns) for a in "ab"), "an agent landed no turn while media ran"
assert slow < float(sys.argv[2]), f"slowest first token {slow:.1f}s >= {sys.argv[2]}s"
PY
tail -c +"$((MARK + 1))" "$LOG" >"$TMP/mix.log"
grep -q "\[gen-yield\] engaged" "$TMP/mix.log" || fail "[3] no [gen-yield] engaged during the mix"
# Two agents are company for each other, so dflash may legitimately hand over: reported, not asserted.
DF=$(grep -c "\[spec-stats\] mode=dflash" "$TMP/mix.log")
[ -z "$MESH" ] || grep -q "decimated [0-9]* -> [0-9]* faces" "$TMP/mix.log" || fail "[3] the mesh job never reached the paint stage"
echo "PASS: [3] mix in ${MIX_SECS}s: every media job valid, image == solo, chat clean and responsive, $DF dflash stat lines"

# [4] Cancel: a music client hangs up mid-denoise; the next turn answers promptly.
sed 's/"seed":7/"seed":8,"stream":true/' "$TMP/music.json" >"$TMP/music_stream.json"
curl -sN -m 1800 "$URL/v1/audio/music-generations" -H 'Content-Type: application/json' -d @"$TMP/music_stream.json" >"$TMP/cancel.sse" &
CUR=$!
for i in $(seq 1 600); do grep -q '"type":"progress"' "$TMP/cancel.sse" 2>/dev/null && break; sleep 0.1; done
kill $CUR 2>/dev/null; wait $CUR 2>/dev/null
python3 "$TMP/turn.py" "$URL" "$CHAT" plain >"$TMP/after.json"
python3 -c "import json,sys; d=json.load(open('$TMP/after.json')); sys.exit(0 if d['ok'] and d['ttft'] < $MAX_TTFT else 1)" ||
  fail "[4] turn after the cancel: $(cat "$TMP/after.json")"
for i in $(seq 1 100); do grep -q "\[music\] generation failed: error.Cancelled" "$LOG" && break; sleep 0.2; done
grep -q "\[music\] generation failed: error.Cancelled" "$LOG" || fail "[4] the abandoned music job was not cancelled"
sed 's/"}$/","stream":true}/' "$TMP/speech.json" >"$TMP/speech_stream.json"
curl -sN -m 1800 "$URL/v1/audio/speech" -H 'Content-Type: application/json' -d @"$TMP/speech_stream.json" >"$TMP/cancel_speech.sse" &
CUR=$!
for i in $(seq 1 600); do grep -q '"type":"progress"' "$TMP/cancel_speech.sse" 2>/dev/null && break; sleep 0.1; done
kill $CUR 2>/dev/null; wait $CUR 2>/dev/null
python3 "$TMP/turn.py" "$URL" "$CHAT" plain >"$TMP/after_speech.json"
python3 -c "import json,sys; d=json.load(open('$TMP/after_speech.json')); sys.exit(0 if d['ok'] and d['ttft'] < $MAX_TTFT else 1)" ||
  fail "[4] turn after the speech cancel: $(cat "$TMP/after_speech.json")"
for i in $(seq 1 100); do grep -q "\[audio\] synthesis cancelled" "$LOG" && break; sleep 0.2; done
grep -q "\[audio\] synthesis cancelled" "$LOG" || fail "[4] the abandoned speech job was not cancelled"
echo "PASS: [4] cancelled music and speech jobs each freed the server (first token $(python3 -c "import json; print(round(json.load(open('$TMP/after.json'))['ttft'], 2), 's /', round(json.load(open('$TMP/after_speech.json'))['ttft'], 2), 's')"))"

# [5] Alive and clean.
python3 "$TMP/turn.py" "$URL" "$CHAT" agent >"$TMP/final.json"
python3 -c "import json,sys; sys.exit(0 if json.load(open('$TMP/final.json'))['ok'] else 1)" || fail "[5] final turn: $(cat "$TMP/final.json")"
! grep -q "\[mlx\]" "$LOG" || fail "[5] MLX error in the log: $(grep -m1 "\[mlx\]" "$LOG")"
echo "ALL PASS: an agent keeps working through image, speech, music${MESH:+ and a textured mesh}"
