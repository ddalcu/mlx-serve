#!/usr/bin/env bash
# POST /v1/audio/transcriptions on Qwen3-ASR (mlx-community's mlx-audio packs):
# discovered under a private --model-dir as a stub advertising "transcription",
# load by id, then every response_format, a forced language, a 48 kHz stereo
# WAV, two clips at once (each keeps its own words), the named 400s, and a
# --no-vision boot that refuses by name. Transcripts are checked by WORD RECALL
# against the script `say` spoke: an audio tower whose rows never reach the
# trunk answers "" (the model hears silence), which no other check can see.
# With Qwen3-ForcedAligner on disk, section 6 rescans it in and checks word and
# segment times, srt/vtt, other scripts, a clip cut past 180 s, and that the
# aligner serves nothing itself (skipped without the aligner pack); with
# Nemotron 3 Diarization too, diarized_json labels three voices taking turns.
#
# Skips without the pack or without macOS `say`. Download with:
#   hf download mlx-community/Qwen3-ASR-0.6B-8bit --local-dir ~/.mlx-serve/models/mlx-community/Qwen3-ASR-0.6B-8bit
#   hf download mlx-community/Qwen3-ForcedAligner-0.6B-8bit --local-dir ~/.mlx-serve/models/mlx-community/Qwen3-ForcedAligner-0.6B-8bit
#   hf download mlx-community/Nemotron-3-Diarization-8bit --local-dir ~/.mlx-serve/models/mlx-community/Nemotron-3-Diarization-8bit
#
# Usage: ASR_MODEL=<dir> ALIGNER_MODEL=<dir> DIARIZER_MODEL=<dir> BINARY=<mlx-serve> ./tests/test_transcriptions.sh [port]
set -uo pipefail
PORT="${1:-11443}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BINARY:-$ROOT/zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
ASR="${ASR_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3-ASR-0.6B-8bit}"
[ -f "$ASR/config.json" ] || { echo "SKIP: no Qwen3-ASR pack at $ASR (set ASR_MODEL)"; exit 0; }
command -v say >/dev/null || { echo "SKIP: needs macOS say to speak the test clips"; exit 0; }

TMP="$(mktemp -d)"
mkdir -p "$TMP/models/mlx-community" "$TMP/home"
ln -s "$ASR" "$TMP/models/mlx-community/$(basename "$ASR")"
ID="mlx-community/$(basename "$ASR")"
LOG="$TMP/server.log"
SRV=""
trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null; rm -rf "$TMP"' EXIT
boot() { # extra flags...
  HOME="$TMP/home" "$BIN" --serve --model-dir "$TMP/models" --port "$PORT" --host 127.0.0.1 "$@" >"$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -5 "$LOG"; exit 1; }
    sleep 1
  done
  echo "FAIL: server not healthy"; exit 1
}
T=/v1/audio/transcriptions
url() { echo "http://127.0.0.1:$PORT$1"; }

SHORT="The quick brown fox jumps over the lazy dog. Speech recognition on Apple Silicon, served natively from Zig."
LONG="Speech recognition has come a long way. Twenty years ago, dictation software needed hours of training on a single voice, and still made mistakes on every other sentence. Today a model with less than one billion parameters can transcribe dozens of languages on a laptop, without sending a single byte of audio to the cloud. That is the promise of running models locally."
say -o "$TMP/short.wav" --data-format=LEI16@16000 "$SHORT"
say -o "$TMP/long48.wav" --data-format=LEI16@48000 --channels=2 "$LONG"

recall() { # spoken heard -> fraction of spoken words heard (case, punctuation and digits-vs-words aside)
  python3 - "$1" "$2" <<'EOF'
import re, sys
norm = lambda s: re.findall(r"[a-z]+", s.lower())
spoken, heard = norm(sys.argv[1]), set(norm(sys.argv[2]))
print(sum(w in heard for w in spoken) / len(spoken))
EOF
}
check_recall() { # label spoken heard
  local r; r=$(recall "$2" "$3")
  python3 -c "import sys; sys.exit(0 if $r >= 0.9 else 1)" || { echo "FAIL: $1 recall $r: '$3'"; exit 1; }
  echo "PASS: $1 (recall $r)"
}
json_field() { python3 -c "import sys,json; print(json.load(sys.stdin)$1)"; }

boot
# 1. Discovery: the stub already says what the pack does.
caps() { # state
  curl -s "$(url /v1/models)" | python3 -c "
import sys,json
m=[x for x in json.load(sys.stdin)['data'] if x['id']=='$ID']
assert m, 'not discovered'
assert m[0]['state']=='$1', m[0]['state']
c=m[0].get('capabilities',[])
assert 'transcription' in c and 'vision' not in c and 'chat' not in c, c
print(c)"
}
c=$(caps unloaded) || { echo "FAIL: stub not discovered with transcription"; exit 1; }
echo "PASS: stub advertises $c"

# 2. Every format; the first request cold-loads the model by id.
J=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID")
TEXT=$(echo "$J" | json_field "['text']") || { echo "FAIL: json format: $J"; tail -20 "$LOG"; exit 1; }
check_recall "json" "$SHORT" "$TEXT"
c=$(caps ready) || { echo "FAIL: ready model lacks transcription"; exit 1; }
echo "PASS: ready model advertises $c"
P=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=text)
[ "$P" = "$TEXT" ] || { echo "FAIL: text format '$P' != json text '$TEXT'"; exit 1; }
echo "PASS: text format is the json text"
V=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=verbose_json -F 'timestamp_granularities[]=segment')
echo "$V" | python3 -c "
import sys,json
v=json.load(sys.stdin)
assert v['task']=='transcribe' and v['language']=='english', v
assert v['text']=='''$TEXT''' and 6.0 < v['duration'] < 8.0, v
s=v['segments']; assert len(s)==1 and s[0]['start']==0 and s[0]['end']==v['duration'], s
" || { echo "FAIL: verbose_json: $V"; exit 1; }
echo "PASS: verbose_json names the detected language and one segment"
F=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F language=en | json_field "['text']")
check_recall "language=en" "$SHORT" "$F"

# 3. 48 kHz stereo is mixed and resampled.
L=$(curl -s "$(url $T)" -F file=@"$TMP/long48.wav" -F model="$ID" | json_field "['text']")
check_recall "48 kHz stereo, 21 s" "$LONG" "$L"

# 4. Two clips at once: each answer holds its own words, not the other's.
curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -o "$TMP/a.json" &
A=$!
curl -s "$(url $T)" -F file=@"$TMP/long48.wav" -F model="$ID" -o "$TMP/b.json" &
B=$!
wait $A $B
check_recall "concurrent short" "$SHORT" "$(json_field "['text']" <"$TMP/a.json")"
check_recall "concurrent long" "$LONG" "$(json_field "['text']" <"$TMP/b.json")"
grep -q "fox" "$TMP/b.json" && { echo "FAIL: the long clip's answer holds the short clip's words"; exit 1; }
echo "PASS: concurrent answers stay apart"

# 5. Named refusals.
b400() { # label needle curl-args...
  local code
  code=$(curl -s -o "$TMP/err.txt" -w "%{http_code}" "$(url $T)" "${@:3}")
  [ "$code" = "400" ] || { echo "FAIL: $1 returned $code (want 400)"; cat "$TMP/err.txt"; exit 1; }
  grep -q "$2" "$TMP/err.txt" || { echo "FAIL: $1 400 does not name '$2'"; cat "$TMP/err.txt"; exit 1; }
  echo "PASS: $1 -> 400"
}
printf 'ID3\x04\x00 not a wav' >"$TMP/fake.mp3"
b400 "no file" "file" -F model="$ID"
b400 "not a WAV" "WAV" -F file=@"$TMP/fake.mp3" -F model="$ID"
b400 "srt without an aligner" "ForcedAligner" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=srt
b400 "stream" "stream" -F file=@"$TMP/short.wav" -F model="$ID" -F stream=true
b400 "unknown language" "language" -F file=@"$TMP/short.wav" -F model="$ID" -F language=xx
b400 "JSON body" "multipart" -H 'Content-Type: application/json' -d "{\"model\":\"$ID\"}"
grep -q "\[mlx\]" "$LOG" && { echo "FAIL: MLX error in the log"; grep "\[mlx\]" "$LOG" | head -3; exit 1; }

# 6. Word times from the forced aligner, rescanned in beside the ASR.
ALIGNER="${ALIGNER_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3-ForcedAligner-0.6B-8bit}"
if [ ! -f "$ALIGNER/config.json" ]; then
  echo "SKIP: word times (no Qwen3-ForcedAligner pack at $ALIGNER; set ALIGNER_MODEL)"
else
  AID="mlx-community/$(basename "$ALIGNER")"
  # An aligner that cannot load (its config, no weights): verbose_json still answers, untimed; srt cannot.
  mkdir -p "$TMP/models/mlx-community/Qwen3-ForcedAligner-broken"
  cp "$ALIGNER/config.json" "$TMP/models/mlx-community/Qwen3-ForcedAligner-broken/"
  curl -s -X POST "$(url /v1/models/rescan)" >/dev/null
  curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=verbose_json -F 'timestamp_granularities[]=word' | python3 -c "
import sys,json
v=json.load(sys.stdin)
assert len(v['segments'])==1 and v['words']==[], v" || { echo "FAIL: verbose_json beside a broken aligner"; exit 1; }
  grep -q "did not load" "$LOG" || { echo "FAIL: no fallback line for the broken aligner"; exit 1; }
  echo "PASS: a broken aligner leaves verbose_json untimed, not failed"
  b400 "srt beside a broken aligner" "did not load" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=srt
  ln -s "$ALIGNER" "$TMP/models/mlx-community/$(basename "$ALIGNER")"
  curl -s -X POST "$(url /v1/models/rescan)" >/dev/null
  state() { curl -s "$(url /v1/models)" | python3 -c "
import sys,json
m=[x for x in json.load(sys.stdin)['data'] if x['id']=='$AID']
assert m, 'aligner not discovered'
c=m[0].get('capabilities',[])
assert c==['alignment'], c
print(m[0]['state'])"; }
  s=$(state) || { echo "FAIL: the aligner stub must advertise only alignment"; exit 1; }
  curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" >/dev/null
  [ "$(state)" = "unloaded" ] || { echo "FAIL: a json transcription loaded the aligner"; exit 1; }
  echo "PASS: the aligner stub advertises alignment and json leaves it unloaded"

  V=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=verbose_json -F 'timestamp_granularities[]=word')
  echo "$V" | python3 -c "
import sys,json
v=json.load(sys.stdin); s=v['segments']; w=v['words']; d=v['duration']
assert len(s)>=2 and s[0]['text'].endswith('.'), s
assert all(0<=x['start']<=x['end']<=d for x in s+w), 'a time outside the clip'
assert all(a['end']<=b['start'] for a,b in zip(s,s[1:])), 'segments overlap'
assert all(a['start']<=b['start'] for a,b in zip(w,w[1:])), 'words go back in time'
assert w[0]['word']=='The' and w[-1]['word'] in ('Zig','zig'), w
assert len(w)==len(v['text'].split()), (len(w), v['text'])
" || { echo "FAIL: verbose_json word times: $V"; exit 1; }
  echo "PASS: verbose_json times every word and splits segments at sentences"
  curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=verbose_json -F 'timestamp_granularities[]=segment' | grep -q '"words"' && { echo "FAIL: words without asking"; exit 1; }
  [ "$(state)" = "ready" ] || { echo "FAIL: the aligner is not resident after timing"; exit 1; }
  echo "PASS: segment granularity carries no words; the aligner loaded on demand"

  SRT=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=srt)
  VTT=$(curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=vtt)
  python3 - "$V" "$SRT" "$VTT" <<'EOF' || { echo "FAIL: subtitles disagree with the segments:"; echo "$SRT"; echo "$VTT"; exit 1; }
import json, re, sys
segs = json.loads(sys.argv[1])["segments"]
srt, vtt = sys.argv[2], sys.argv[3]
def cues(text, sep):
    t = re.findall(r"(\d\d):(\d\d):(\d\d)" + sep + r"(\d{3}) --> (\d\d):(\d\d):(\d\d)" + sep + r"(\d{3})\n(.+)", text)
    return [(int(a) * 3600 + int(b) * 60 + int(c) + int(d) / 1000, int(e) * 3600 + int(f) * 60 + int(g) + int(h) / 1000, x) for a, b, c, d, e, f, g, h, x in t]
assert vtt.startswith("WEBVTT\n\n") and srt.startswith("1\n")
for got in (cues(srt, ","), cues(vtt, r"\.")):
    assert len(got) == len(segs), got
    for (s, e, x), seg in zip(got, segs):
        assert abs(s - seg["start"]) < 0.002 and abs(e - seg["end"]) < 0.002 and x == seg["text"], (s, e, x, seg)
EOF
  echo "PASS: srt and vtt cues are the verbose_json segments"

  voice() { say -v '?' | sed -n "s/^\(.*[^ ]\)  *$1 .*/\1/p" | head -1; }
  ZH=$(voice zh_CN); KO=$(voice ko_KR); AR=$(voice ar_001)
  words() { curl -s "$(url $T)" -F file=@"$1" -F model="$ID" -F response_format=verbose_json -F 'timestamp_granularities[]=word'; }
  if [ -n "$ZH" ]; then
    say -v "$ZH" -o "$TMP/zh.wav" --data-format=LEI16@16000 "今天天气很好，我们一起去公园散步吧。"
    words "$TMP/zh.wav" | python3 -c "
import sys,json
v=json.load(sys.stdin); w=[x['word'] for x in v['words']]
han=[c for c in v['text'] if '一'<=c<='鿿']
assert v['language']=='chinese' and w==han, (w, v['text'])" || { echo "FAIL: Chinese is timed one character at a time"; exit 1; }
    echo "PASS: Chinese words are its characters"
  fi
  if [ -n "$KO" ]; then
    say -v "$KO" -o "$TMP/ko.wav" --data-format=LEI16@16000 "오늘은 날씨가 정말 좋네요. 친구와 함께 공원을 산책했습니다."
    words "$TMP/ko.wav" | python3 -c "
import sys,json,re
v=json.load(sys.stdin); w=[x['word'] for x in v['words']]
assert v['language']=='korean' and w==[re.sub(r'[.,!?]','',t) for t in v['text'].split()], (w, v['text'])" || { echo "FAIL: Korean is timed by spaced word"; exit 1; }
    echo "PASS: Korean words are its spaced words"
  fi
  if [ -n "$AR" ]; then
    say -v "$AR" -o "$TMP/ar.wav" --data-format=LEI16@16000 "مرحبا، الطقس جميل اليوم."
    words "$TMP/ar.wav" | python3 -c "
import sys,json
v=json.load(sys.stdin)
assert v['language']=='arabic' and len(v['segments'])==1 and v['words']==[], v
assert v['segments'][0]['start']==0 and v['segments'][0]['end']==v['duration']" || { echo "FAIL: a language the aligner lacks is one segment"; exit 1; }
    echo "PASS: Arabic (no aligner support) is one segment with no words"
  fi

  # Past 180 s the clip is cut where it is quietest, each piece transcribed and timed.
  LONG10=""; for i in 1 2 3 4 5 6 7 8 9 10; do LONG10="$LONG10 $LONG"; done
  say -o "$TMP/long10.wav" --data-format=LEI16@16000 "$LONG10"
  before=$(grep -c "  aligned " "$LOG")
  curl -s "$(url $T)" -F file=@"$TMP/long10.wav" -F model="$ID" -F response_format=verbose_json -F 'timestamp_granularities[]=word' | python3 -c "
import sys,json
v=json.load(sys.stdin); w=v['words']; d=v['duration']
assert d>180, d
assert all(a['start']<=b['start'] for a,b in zip(w,w[1:])), 'words go back in time across the cut'
assert w[-1]['end']>d-10 and all(x['end']<=d for x in w), (w[-1], d)" || { echo "FAIL: a clip past 180 s"; exit 1; }
  [ $(( $(grep -c "  aligned " "$LOG") - before )) -eq 2 ] || { echo "FAIL: a 210 s clip is not two pieces"; exit 1; }
  echo "PASS: a 210 s clip is two pieces, timed end to end"

  # 6b. Speakers from Nemotron 3 Diarization, rescanned in beside the aligner.
  DIARIZER="${DIARIZER_MODEL:-$HOME/.mlx-serve/models/mlx-community/Nemotron-3-Diarization-8bit}"
  pick_voice() { # preferred name, else the first voice of a locale
    say -v '?' | grep -q "^$1 " && { echo "$1"; return; }
    say -v '?' | sed -n "s/^\(.*[^ ]\)  *$2 .*/\1/p" | head -1
  }
  V1=$(pick_voice Daniel en_GB); V2=$(pick_voice "Samantha (English (US))" en_US); V3=$(pick_voice Rishi en_IN)
  if [ ! -f "$DIARIZER/config.json" ]; then
    echo "SKIP: speakers (no Nemotron 3 Diarization pack at $DIARIZER; set DIARIZER_MODEL)"
  elif [ -z "$V1" ] || [ -z "$V2" ] || [ -z "$V3" ]; then
    echo "SKIP: speakers (needs an en_GB, an en_US and an en_IN voice for say)"
  else
    b400 "diarized_json without a diarizer" "Nemotron-3-Diarization" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=diarized_json
    b400 "known speakers" "known speakers" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=diarized_json -F 'known_speaker_names[]=agent'
    # A diarizer that cannot load (its config, no weights): one speaker, not a failure.
    mkdir -p "$TMP/models/mlx-community/Nemotron-3-Diarization-broken"
    cp "$DIARIZER/config.json" "$TMP/models/mlx-community/Nemotron-3-Diarization-broken/"
    curl -s -X POST "$(url /v1/models/rescan)" >/dev/null
    curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=diarized_json | python3 -c "
import sys,json
v=json.load(sys.stdin)
assert v['segments'] and {s['speaker'] for s in v['segments']}=={'A'}, v" || { echo "FAIL: diarized_json beside a broken diarizer"; exit 1; }
    grep -q "diarizer .* did not load" "$LOG" || { echo "FAIL: no fallback line for the broken diarizer"; exit 1; }
    echo "PASS: a broken diarizer leaves one speaker, not a failure"
    DID="mlx-community/$(basename "$DIARIZER")"
    ln -s "$DIARIZER" "$TMP/models/mlx-community/$(basename "$DIARIZER")"
    curl -s -X POST "$(url /v1/models/rescan)" >/dev/null
    dstate() { curl -s "$(url /v1/models)" | python3 -c "
import sys,json
m=[x for x in json.load(sys.stdin)['data'] if x['id']=='$DID']
assert m, 'diarizer not discovered'
assert m[0].get('capabilities')==['diarization'], m[0].get('capabilities')
print(m[0]['state'])"; }
    s=$(dstate) || { echo "FAIL: the diarizer stub must advertise only diarization"; exit 1; }
    curl -s "$(url $T)" -F file=@"$TMP/short.wav" -F model="$ID" -F response_format=verbose_json >/dev/null
    [ "$(dstate)" = "unloaded" ] || { echo "FAIL: verbose_json loaded the diarizer"; exit 1; }
    echo "PASS: the diarizer stub advertises diarization and verbose_json leaves it unloaded"

    # Three voices taking turns, two of them twice; where each turn falls is known.
    python3 - "$TMP" "$V1" "$V2" "$V3" <<'EOF' || { echo "FAIL: could not build the conversation"; exit 1; }
import json, subprocess, sys, wave
tmp, voices = sys.argv[1], sys.argv[2:]
turns = [(0, "Good morning everyone, let's go through the release plan for this week."),
         (1, "Morning. The beta build is ready, and the installer is finally signed."),
         (2, "Great news. Did the crash on older machines get fixed before the build?"),
         (0, "It did. Let's ship it to the testers on Thursday afternoon."),
         (1, "I will send the notes to the support team tomorrow morning."),
         (2, "Perfect, then the documentation review can happen on Friday.")]
pcm, truth, rate = b"", [], 16000
for i, (who, text) in enumerate(turns):
    out = f"{tmp}/turn{i}.wav"
    subprocess.run(["say", "-v", voices[who], "-o", out, "--data-format=LEI16@16000", text], check=True)
    with wave.open(out) as w:
        frames = w.readframes(w.getnframes())
    pcm += b"\0\0" * int(0.6 * rate)
    start = len(pcm) // 2 / rate
    pcm += frames
    truth.append({"voice": who, "start": start, "end": len(pcm) // 2 / rate})
pcm += b"\0\0" * int(0.6 * rate)
with wave.open(f"{tmp}/conv.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
json.dump(truth, open(f"{tmp}/conv_truth.json", "w"))
EOF
    before=$(grep -c "  diarized " "$LOG")
    D=$(curl -s "$(url $T)" -F file=@"$TMP/conv.wav" -F model="$ID" -F response_format=diarized_json)
    echo "$D" | python3 -c "
import sys,json
v=json.load(sys.stdin); s=v['segments']; truth=json.load(open('$TMP/conv_truth.json'))
assert v['task']=='transcribe' and v['duration']>truth[-1]['end'] and v['text'], v
assert [x['id'] for x in s]==['seg_%03d'%(i+1) for i in range(len(s))] and all(x['type']=='transcript.text.segment' for x in s), s
assert all(a['start']<=b['start'] and x['start']<=x['end'] for a,b,x in zip(s,s[1:],s)), 'segments out of order'
seen=[]
for x in s:
    if x['speaker'] not in seen: seen.append(x['speaker'])
assert seen==[chr(65+i) for i in range(len(seen))], ('letters not in order of first speech', seen)
pairs=[]
for x in s:
    mid=(x['start']+x['end'])/2
    t=[u['voice'] for u in truth if u['start']-0.3<=mid<=u['end']+0.3]
    if t: pairs.append((t[0], x['speaker']))
best={vo: max(set(l for w,l in pairs if w==vo), key=[l for w,l in pairs if w==vo].count) for vo in {w for w,_ in pairs}}
assert len(set(best.values()))==3, ('three voices, three letters', best, pairs)
right=sum(best[w]==l for w,l in pairs)/len(pairs)
assert right>=0.9, (right, pairs)
print(len(s), 'segments', seen, right)
" || { echo "FAIL: diarized_json speakers: $D"; exit 1; }
    [ $(( $(grep -c "  diarized " "$LOG") - before )) -eq 1 ] || { echo "FAIL: the clip was not diarized once"; exit 1; }
    [ "$(dstate)" = "ready" ] || { echo "FAIL: the diarizer is not resident after labelling"; exit 1; }
    echo "PASS: three voices get three letters, in order of first speech, on their own turns"
    code=$(curl -s -o "$TMP/err.txt" -w "%{http_code}" "$(url /v1/chat/completions)" -H 'Content-Type: application/json' -d "{\"model\":\"$DID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
    [ "$code" = "400" ] && grep -q "diarized_json" "$TMP/err.txt" || { echo "FAIL: chat on the diarizer returned $code"; cat "$TMP/err.txt"; exit 1; }
    echo "PASS: chat on the diarizer -> 400"
    b400 "transcribing with the diarizer" "diarized_json" -F file=@"$TMP/short.wav" -F model="$DID"
  fi

  code=$(curl -s -o "$TMP/err.txt" -w "%{http_code}" "$(url /v1/chat/completions)" -H 'Content-Type: application/json' -d "{\"model\":\"$AID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
  [ "$code" = "400" ] && grep -q "Qwen3-ASR" "$TMP/err.txt" || { echo "FAIL: chat on the aligner returned $code"; cat "$TMP/err.txt"; exit 1; }
  echo "PASS: chat on the aligner -> 400"
  b400 "transcribing with the aligner" "Qwen3-ASR" -F file=@"$TMP/short.wav" -F model="$AID"
  code=$(curl -s -o /dev/null -w "%{http_code}" "$(url /v1/chat/completions)" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}')
  [ "$code" = "503" ] || { echo "FAIL: a model-less chat answered $code: an audio model became the default"; exit 1; }
  echo "PASS: neither audio model became the default chat model"
  grep -q "\[mlx\]" "$LOG" && { echo "FAIL: MLX error in the log"; grep "\[mlx\]" "$LOG" | head -3; exit 1; }
  rm "$TMP/models/mlx-community/$(basename "$ALIGNER")"
fi

# 7. Without its tower the model cannot transcribe, and says why.
kill $SRV; wait $SRV 2>/dev/null
boot --no-vision
b400 "--no-vision" "Qwen3-ASR" -F file=@"$TMP/short.wav" -F model="$ID"
echo "ALL PASS"
