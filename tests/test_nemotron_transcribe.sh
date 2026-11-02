#!/bin/bash
# Nemotron 3.5 Transcribe live end-to-end (the nemotron_transcribe plugin):
#
#   NEMOTRON_TRANSCRIBE_MODEL=/path/to/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-Serve-8bit ./tests/test_nemotron_transcribe.sh
#   (pack from scripts/convert_nemotron_transcribe.py; default: the first pack found on disk)
#
#   [1] /v1/models: audio input and transcription     [5] chat input_audio: a spoken question's transcript
#   [2] /v1/audio/transcriptions hears macOS `say`    [6] speech translation
#   [3] a long clip across the 40 s window            [7] named 400s: language + prompt, a non-WAV file
#   [4] chat input_audio with the card's recipe       [8] speaker tags on two voices
#   [9] verbose_json word times from its CTC head     [10] diarized_json from its own speaker expert
#
# The clips come from macOS `say` + ffmpeg (SKIP without them): no fixture to keep in sync.
# Hermetic counterpart: src/plugins/nemotron_transcribe/proof_test.zig (the tower over a tiny pack).

set -euo pipefail
cd "$(dirname "$0")/.."
source tests/_lib_models.sh
source tests/lib/server.sh

command -v say >/dev/null && command -v ffmpeg >/dev/null || { echo "SKIP: needs macOS say and ffmpeg"; exit 0; }
MODEL="${NEMOTRON_TRANSCRIBE_MODEL:-$(find_model NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-Serve-8bit)}"
if [ -z "$MODEL" ]; then echo "SKIP: no Nemotron Transcribe pack on disk (set NEMOTRON_TRANSCRIBE_MODEL)"; exit 0; fi
model_fits "$MODEL" || { echo "SKIP: $(model_gb "$MODEL") GB > $(max_model_gb) GB"; exit 0; }

PORT="${NEMOTRON_TRANSCRIBE_TEST_PORT:-11381}"
BASE="http://127.0.0.1:$PORT"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
SCRATCH=$(mktemp -d "$HOME/claude-tmp/nemotron_transcribe_test.XXXXXX")
LOG="$SCRATCH/server.log"

say_wav() { # voice out text...
  local voice="$1" out="$2"; shift 2
  say -v "$voice" -o "$SCRATCH/clip.aiff" "$*"
  ffmpeg -y -loglevel error -i "$SCRATCH/clip.aiff" -ar 16000 -ac 1 -sample_fmt s16 "$out"
}
say_wav Samantha "$SCRATCH/en.wav" "The quick brown fox jumps over the lazy dog. Nemotron transcribes speech on Apple silicon."
say_wav Samantha "$SCRATCH/question.wav" "What is the capital of France?"
say_wav Samantha "$SCRATCH/dog.wav" "The dog is sleeping in the garden."
say_wav Samantha "$SCRATCH/long.wav" "Chapter one. It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him. The hallway smelt of boiled cabbage and old rag mats. At one end of it a coloured poster, too large for indoor display, had been tacked to the wall."
say_wav Samantha "$SCRATCH/a.wav" "Hello, how are you doing today? I wanted to ask about the meeting tomorrow."
say_wav Daniel "$SCRATCH/b.wav" "I am fine thank you. The meeting is at three in the afternoon."
ffmpeg -y -loglevel error -i "$SCRATCH/a.wav" -i "$SCRATCH/b.wav" -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1" "$SCRATCH/two.wav"

HOME="$SCRATCH" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level debug > "$LOG" 2>&1 &
SERVER_PID=$!
cleanup() { srv_stop "$SERVER_PID"; rm -rf "$SCRATCH"; }
trap cleanup EXIT
srv_wait_health "$BASE" 600 0.5 "$SERVER_PID" || { echo "FAIL: server did not come up"; tail -20 "$LOG"; exit 1; }

# A pack converted without the CTC head (mlx-audio's) transcribes but cannot time words.
HAS_CTC=$(grep -c "ctc_timestamp" "$MODEL/model.safetensors.index.json" 2>/dev/null || true)

python3 -I - "$BASE" "$LOG" "$SCRATCH" "${HAS_CTC:-0}" <<'PY'
import base64, json, re, sys, urllib.request, urllib.error, uuid
base, log, scratch, has_ctc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) > 0
fails = []
def ok(name, cond, got=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n  got: {str(got)[:400]}"))
    if not cond: fails.append(name)
def post(path, body, ctype="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    r = urllib.request.Request(base + path, data, {"content-type": ctype})
    try:
        resp = urllib.request.urlopen(r, timeout=900); return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
def form(fields, wav):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + wav + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return post("/v1/audio/transcriptions", b"".join(parts), f"multipart/form-data; boundary={boundary}")
def transcribe(name, **fields):
    st, b = form(fields, open(f"{scratch}/{name}.wav", "rb").read())
    return st, (json.loads(b) if b.startswith(b"{") else b.decode(errors="replace"))
def said(name): return {"type": "input_audio", "input_audio": {"data": base64.b64encode(open(f"{scratch}/{name}.wav", "rb").read()).decode(), "format": "wav"}}
def chat(content, **kw):
    st, b = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "max_tokens": 400, "temperature": 0,
                                          "chat_template_kwargs": {"enable_thinking": False}, **kw})
    return st, (json.loads(b) if b.startswith(b"{") else b)
def words(s): return re.findall(r"[a-z0-9]+", s.lower())
def recall(heard, said_text):
    want, got = words(said_text), set(words(heard))
    return sum(w in got for w in want) / len(want)

EN = "The quick brown fox jumps over the lazy dog. Nemotron transcribes speech on Apple silicon."
row = [m for m in json.load(urllib.request.urlopen(base + "/v1/models"))["data"] if m.get("loaded")][0]
caps = row.get("capabilities") or []
ok("[1] /v1/models: audio input and transcription", "transcription" in caps and "audio" in (row.get("input_modalities") or caps), row)

st, r = transcribe("en", response_format="json")
ok("[2] transcription hears the clip", st == 200 and recall(r["text"], EN) >= 0.85 and "fox" in r["text"].lower(), r)

LONG = "Chapter one. It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith slipped quickly through the glass doors of Victory Mansions"
st, r = transcribe("long", response_format="json")
ok("[3] a 22 s clip across several windows", st == 200 and recall(r["text"], LONG) >= 0.85 and "cabbage" in r["text"].lower(), r)

recipe = "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only."
st, r = chat([{"type": "text", "text": recipe}, said("en")])
text = r["choices"][0]["message"]["content"] if st == 200 else ""
ok("[4] chat input_audio, the card's recipe", st == 200 and recall(text, EN) >= 0.85, r)

st, r = chat([{"type": "text", "text": "Transcribe this audio."}, said("question")])
text = r["choices"][0]["message"]["content"] if st == 200 else ""
ok("[5] chat input_audio, a spoken question", st == 200 and "capital" in text.lower() and "france" in text.lower(), r)

st, r = chat([{"type": "text", "text": "Please translate the given speech to German."}, said("dog")])
text = r["choices"][0]["message"]["content"] if st == 200 else ""
ok("[6] speech translation to German", st == 200 and "hund" in text.lower(), r)

st, r = transcribe("en", language="English", prompt="Nemotron")
ok("[7a] language and prompt together is a named 400", st == 400 and "not both" in json.dumps(r), (st, r))
st, b = form({}, b"not a wav")
ok("[7b] a non-WAV file is a named 400", st == 400, (st, b))

spk = "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
st, r = chat([{"type": "text", "text": spk}, said("two")], skip_special_tokens=False)
text = r["choices"][0]["message"]["content"] if st == 200 else ""
ok("[8] two voices carry speaker tags", st == 200 and "<spk:0>" in text and "<spk:1>" in text, text)

if not has_ctc:
    st, r = transcribe("en", response_format="verbose_json", **{"timestamp_granularities[]": "word"})
    ok("[9] without the CTC head verbose_json still answers, its words untimed", st == 200 and not r.get("words") and r.get("text"), (st, r))
    st, r = transcribe("en", response_format="srt")
    ok("[9b] srt names the missing head", st == 400 and "CTC" in json.dumps(r), (st, r))
    st, r = transcribe("two", response_format="diarized_json")
    ok("[10] diarized_json answers with a speaker per piece", st == 200 and r.get("segments"), (st, r))
    sys.exit(1 if fails else 0)

st, r = transcribe("en", response_format="verbose_json", **{"timestamp_granularities[]": "word"})
words = r.get("words", []) if st == 200 else []
mono = all(w["start"] <= w["end"] and (i == 0 or words[i - 1]["start"] <= w["start"]) for i, w in enumerate(words))
fox = [w for w in words if w["word"].lower().strip(".,") == "fox"]
ok("[9] verbose_json times every word, in order, inside the clip", st == 200 and len(words) >= 12 and mono and words[-1]["end"] < 12
   and fox and 0.4 < fox[0]["start"] < 2.0, (st, words[:6], words[-3:]))
ok("[9b] srt answers with its own times", transcribe("en", response_format="srt")[0] == 200)

import wave
with wave.open(f"{scratch}/a.wav") as w: first_len = w.getnframes() / w.getframerate()
st, r = transcribe("two", response_format="diarized_json")
segs = r.get("segments", []) if st == 200 else []
speakers = {s.get("speaker") for s in segs} - {None}
ok("[10] diarized_json hears two voices without a companion model", st == 200 and len(speakers) >= 2, (st, r if st != 200 else sorted(map(str, speakers))))
ok("[10b] speaker turns carry no <spk:> tags in their text", st == 200 and not any("<spk" in s["text"] for s in segs), segs[:3])
before = {s["speaker"] for s in segs if s["end"] <= first_len + 0.3}
after = {s["speaker"] for s in segs if s["start"] >= first_len - 0.3}
ok("[10c] the first voice speaks until its clip ends, the second after", len(before) == 1 and len(after) == 1 and before != after, (first_len, [(s["speaker"], round(s["start"], 2), round(s["end"], 2)) for s in segs]))
st, r = transcribe("two", response_format="verbose_json", **{"timestamp_granularities[]": "word"})
words = r.get("words", []) if st == 200 else []
ok("[10d] words are timed across both clips, in order", len(words) > 20 and all(words[i]["start"] <= words[i + 1]["start"] + 0.001 for i in range(len(words) - 1)) and words[-1]["end"] > first_len, (st, words[-3:]))

sys.exit(1 if fails else 0)
PY
