#!/bin/bash
# LFM3 family live end-to-end (LiquidAI LFM3 1.7B / VL / Audio / Omni, the lfm3 plugin):
#
#   LFM3_MODEL=/path/to/LiquidAI/LFM3-Omni-2B-MLX-8bit ./tests/test_lfm3.sh
#   (default: the first of Omni, Audio, VL, 1.7B 8-bit found on disk)
#
#   [1] text answer                          [6] spoken chat reply: OpenAI message.audio, heard back by ASR
#   [2] tool call, no markup in content      [7] streamed spoken reply: transcript + pcm16 deltas, frames logged
#   [3] image question (VL, Omni)            [8] history turn naming an audio id replays; an unknown id is a 400
#   [4] /v1/audio/speech, plain and stream    [9] named 400s; a spoken tool round trip (or its 400)
#   [5] /v1/audio/transcriptions hears [4]  [10] /v1/models: speech, output_modalities, voices
#
# A model's checks follow what it advertises: speech ones need output_modalities audio, [3] needs image input.
# The WAV the ASR and voice-chat checks hear is the server's own TTS ([4]): no fixture to keep in sync.
# Hermetic counterparts: src/plugins/lfm3/proof_test.zig + served_test.zig (tiny fixtures vs mlx-vlm),
# engines/mlx/generate/speech.zig (the slot machine), host/api/speech.zig (the request and the audio).

set -euo pipefail
cd "$(dirname "$0")/.."
source tests/_lib_models.sh
source tests/lib/server.sh

MODEL="${LFM3_MODEL:-$(find_model LiquidAI/LFM3-Omni-2B-MLX-8bit LiquidAI/LFM3-Audio-2B-MLX-8bit LiquidAI/LFM3-VL-2B-MLX-8bit LiquidAI/LFM3-1.7B-MLX-8bit)}"
if [ -z "$MODEL" ]; then echo "SKIP: no LFM3 pack on disk (set LFM3_MODEL)"; exit 0; fi
model_fits "$MODEL" || { echo "SKIP: $(model_gb "$MODEL") GB > $(max_model_gb) GB"; exit 0; }

PORT="${LFM3_TEST_PORT:-11379}"
BASE="http://127.0.0.1:$PORT"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
SCRATCH=$(mktemp -d "$HOME/claude-tmp/lfm3_test.XXXXXX")
LOG="$SCRATCH/server.log"

HOME="$SCRATCH" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level debug > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { srv_stop "$SERVER_PID"; rm -rf "$SCRATCH"; }
trap cleanup EXIT
srv_wait_health "$BASE" 300 0.5 "$SERVER_PID" || { echo "FAIL: server did not come up"; tail -20 "$LOG"; exit 1; }

python3 -I - "$BASE" "$LOG" <<'PY'
import base64, json, re, sys, urllib.request, urllib.error
base, log = sys.argv[1], sys.argv[2]
fails = []
def ok(name, cond, got=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n  got: {str(got)[:300]}"))
    if not cond: fails.append(name)
def post(path, body, ctype="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    r = urllib.request.Request(base + path, data, {"content-type": ctype})
    try:
        resp = urllib.request.urlopen(r, timeout=600); return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
def chat(messages, **kw):
    st, b = post("/v1/chat/completions", {"messages": messages, "max_tokens": 300, "temperature": 0, **kw})
    return st, (json.loads(b) if b.startswith(b"{") else b)
def said(wav): return {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}
def words(s): return re.findall(r"[a-z0-9]+", s.lower())

row = [m for m in json.load(urllib.request.urlopen(base + "/v1/models"))["data"] if m.get("loaded")][0]
speaks = "audio" in (row.get("output_modalities") or [])
sees = "image" in row.get("input_modalities", [])
print(f"model {row['id']}: speaks={speaks} sees={sees}")

st, r = chat([{"role": "user", "content": "What is 17 times 3? Answer with the number only."}])
ok("[1] text answer", st == 200 and "51" in r["choices"][0]["message"]["content"], r)

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
st, r = chat([{"role": "user", "content": "What is the weather in Paris? Use the tool."}], tools=tools)
m = r["choices"][0]["message"] if st == 200 else {}
calls = m.get("tool_calls") or []
ok("[2] tool call, no markup in content", calls and calls[0]["function"]["name"] == "get_weather"
   and "<tool_call>" not in (m.get("content") or ""), r)

if sees:
    img = base64.b64encode(open("tests/fixtures/robot.png", "rb").read()).decode()
    st, r = chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                                               {"type": "text", "text": "What is in this picture? One sentence."}]}])
    ok("[3] image question", st == 200 and "robot" in r["choices"][0]["message"]["content"].lower(), r)

if speaks:
    line = "Good morning. The train to Cambridge leaves at nine."
    st, wav = post("/v1/audio/speech", {"input": line, "voice": "british-english-male"})
    ok("[4] /v1/audio/speech", st == 200 and wav[:4] == b"RIFF" and len(wav) > 44 + 48000, wav[:200])
    # The app's voices (Kokoro, Qwen3-TTS, LFM3) all ask with stream: true and read the SSE `complete` event.
    st, body = post("/v1/audio/speech", {"input": "Hello there.", "voice": "american-english-female", "stream": True})
    events = [json.loads(l[6:]) for l in body.decode(errors="replace").splitlines() if l.startswith("data: {")]
    done = [e for e in events if e.get("type") == "complete"]
    ok("[4] /v1/audio/speech streams like every voice", st == 200 and done and done[0].get("format") == "wav"
       and base64.b64decode(done[0]["data"])[:4] == b"RIFF", body[:200])
    form = (b"--X\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n"
            b"--X\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            + wav + b"\r\n--X--\r\n")
    st, body = post("/v1/audio/transcriptions", form, "multipart/form-data; boundary=X")
    heard = json.loads(body).get("text", "") if st == 200 else body
    ok("[5] transcription hears the speech", st == 200 and words(heard) == words(line), heard)

    st, r = chat([{"role": "user", "content": [said(wav), {"type": "text", "text": "Where does the train go?"}]}],
                 modalities=["text", "audio"], audio={"voice": "default", "format": "wav"})
    a = r["choices"][0]["message"].get("audio") or {} if st == 200 else {}
    reply = base64.b64decode(a.get("data", ""))
    ok("[6] spoken reply is message.audio", st == 200 and r["choices"][0]["message"]["content"] is None
       and reply[:4] == b"RIFF" and len(reply) > 44 + 24000 and a.get("transcript"), r if st != 200 else a.get("transcript"))
    if reply:
        st2, r2 = chat([{"role": "system", "content": "Perform ASR in English."}, {"role": "user", "content": [said(reply)]}])
        back = r2["choices"][0]["message"]["content"] if st2 == 200 else r2
        common = set(words(back)) & set(words(a["transcript"]))
        ok("[6] its speech says its transcript", len(common) >= max(2, len(set(words(a["transcript"]))) // 2), (back, a["transcript"]))
        st, r = chat([{"role": "user", "content": [said(wav)]}, {"role": "assistant", "audio": {"id": a["id"]}},
                      {"role": "user", "content": "Repeat the city you named, one word."}])
        ok("[8] a history turn names its audio id", st == 200 and "cambridge" in r["choices"][0]["message"]["content"].lower(), r)
    st, r = chat([{"role": "assistant", "audio": {"id": "audio_00000000000000000000000000000000"}}, {"role": "user", "content": "hi"}])
    ok("[8] an unknown audio id is a named 400", st == 400 and b"audio.id" in (r if isinstance(r, bytes) else json.dumps(r).encode()), r)

    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps({
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 300, "stream": True,
        "modalities": ["text", "audio"], "audio": {"format": "pcm16"}}).encode(), {"content-type": "application/json"})
    pcm, transcript, finish = b"", "", None
    for raw in urllib.request.urlopen(req, timeout=600):
        s = raw.decode().strip()
        if not s.startswith("data: ") or s == "data: [DONE]": continue
        for ch in json.loads(s[6:]).get("choices", []):
            finish = ch.get("finish_reason") or finish
            d = (ch.get("delta") or {}).get("audio") or {}
            pcm += base64.b64decode(d.get("data", "")); transcript += d.get("transcript", "")
    frames = re.findall(r"tokens streamed, (\d+) frames", open(log).read())
    ok("[7] streamed spoken reply", finish == "stop" and len(pcm) > 24000 and transcript.strip() and frames and int(frames[-1]) > 0,
       (finish, len(pcm), transcript, frames))

    def status(body, path="/v1/chat/completions"): return post(path, body)[0]
    hi = [{"role": "user", "content": "hi"}]
    # The card's spoken recipe opens no think block: a reply's prompt is the same whatever `enable_thinking` says.
    def prompt_tokens(think):
        st, r = chat([{"role": "user", "content": "Name three colors."}], max_tokens=40, enable_thinking=think,
                     modalities=["text", "audio"], audio={"voice": "default", "format": "wav"})
        return r["usage"]["prompt_tokens"] if st == 200 else r
    ok("[9] a spoken reply's prompt ignores enable_thinking", prompt_tokens(True) == prompt_tokens(False), (prompt_tokens(True), prompt_tokens(False)))
    ok("[9] a named chat voice is a 400 (the reply speaks in the model's one voice)",
       status({"messages": hi, "modalities": ["audio"], "audio": {"voice": "british-english-male"}}) == 400)
    ok("[9] mp3 is a named 400", status({"messages": hi, "modalities": ["audio"], "audio": {"format": "mp3"}}) == 400)
    ok("[9] a wav stream is a named 400", status({"messages": hi, "stream": True, "modalities": ["audio"], "audio": {"format": "wav"}}) == 400)
    if "speech_tools" in row["capabilities"]:
        # The card's tools recipe, streamed: a call arrives as `tool_calls`; with its result the answer is spoken.
        def stream_turn(messages):
            req = urllib.request.Request(base + "/v1/chat/completions", json.dumps({
                "messages": messages, "tools": tools, "max_tokens": 400, "temperature": 0, "stream": True,
                "modalities": ["text", "audio"], "audio": {"format": "pcm16"}}).encode(), {"content-type": "application/json"})
            calls, pcm, finish = [], b"", None
            for raw in urllib.request.urlopen(req, timeout=600):
                s = raw.decode().strip()
                if not s.startswith("data: {"): continue
                for ch in json.loads(s[6:]).get("choices", []):
                    finish = ch.get("finish_reason") or finish
                    d = ch.get("delta") or {}
                    calls += d.get("tool_calls") or []
                    pcm += base64.b64decode((d.get("audio") or {}).get("data", ""))
            return calls, pcm, finish
        st, wq = post("/v1/audio/speech", {"input": "What's the weather like in Paris right now?", "voice": "american-english-male"})
        q = [{"role": "user", "content": [said(wq)]}]
        calls, pcm, finish = stream_turn(q)
        ok("[9] a streamed spoken request calls its tool", finish == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather"
           and not pcm, (finish, calls, len(pcm)))
        if calls:
            q += [{"role": "assistant", "tool_calls": [{"id": calls[0]["id"], "type": "function", "function": calls[0]["function"]}]},
                  {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"city": "Paris", "temp_c": 18, "sky": "light rain"})}]
            calls, pcm, finish = stream_turn(q)
            ok("[9] ...and speaks the answer from its result", finish == "stop" and not calls and len(pcm) > 24000, (finish, calls, len(pcm)))
    else:
        # A model whose card answers tools in text (Omni) says so by name rather than speaking a guess.
        ok("[9] speech + tools on a model that answers tools in text is a named 400",
           status({"messages": hi, "tools": tools, "modalities": ["audio"], "audio": {"format": "wav"}}) == 400)
    form = (b"--X\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nnames\r\n"
            b"--X\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            + wav + b"\r\n--X--\r\n")
    ok("[9] an ASR prompt is a named 400", post("/v1/audio/transcriptions", form, "multipart/form-data; boundary=X")[0] == 400)
    ok("[10] models row speaks", "speech" in row["capabilities"] and "transcription" in row["capabilities"]
       and "audio" in row["input_modalities"] and len(row.get("voices", [])) == 38, row)
else:
    st, r = chat([{"role": "user", "content": "hi"}], modalities=["text", "audio"])
    ok("[9] a model that does not speak refuses audio by name", st == 400, r)

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
PY
