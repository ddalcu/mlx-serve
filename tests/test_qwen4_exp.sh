#!/usr/bin/env bash
# Qwen3.8-Flash-Next (qwen4_exp) live end-to-end on the converted pack.
# Boots the pack, then: architecture advertised, a short greedy answer, a tool
# call + round-trip, streaming delta cleanliness, and a prompt past the QSA
# budget (2048 tokens) — asserted through the server's own
# `[qsa] sparse attention engaged` line, since a dense fallback answers
# plausibly too. [7] sends an image (tower + M-RoPE engagement lines) and
# SKIPs when the pack ships no `model-vision.safetensors`. [11] reboots
# `--no-vision` (tower absent, text works, media 400s by name). SKIPs without the pack.
#   QWEN4_MODEL=<pack dir> ./tests/test_qwen4_exp.sh [port]
set -u
MODEL="${QWEN4_MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit}"
PORT="${1:-11411}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
LOG="$HOME/claude-tmp/qwen4-live/server-$PORT.log"
mkdir -p "$(dirname "$LOG")"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no pack at $MODEL"; exit 0; }
[ -f "$MODEL/ngram_table.bin" ] || { echo "SKIP: pack has no ngram_table.bin"; exit 0; }
pass=0; fail=0
# Tie-aware equivalence (test_mtp_equivalence.sh bar): prints 1 when `other`
# equals the serial greedy answer for body `$1`, or first diverges at a token
# whose serial top-2 gap is <= 0.15 nats.
same_or_tie() {
  local body="$1" other="$2"
  local lp; lp=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); d['logprobs']=True; d['top_logprobs']=2; d['enable_mtp']=False; print(json.dumps(d))")
  curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$lp" | python3 -c "
import sys,json; d=json.load(sys.stdin); other=sys.argv[1]; acc=''; ok=1
for n,e in enumerate(d['choices'][0]['logprobs']['content'][:30]):
    if not other.startswith((acc+e['token']).lstrip()):  # content is lead-trimmed, tokens are not
        t=e['top_logprobs']; gap=(t[0]['logprob']-t[1]['logprob']) if len(t)>1 else 99
        print('  diverged at token %d, serial top-2 gap %.3f nats' % (n, gap), file=sys.stderr)
        ok=1 if gap <= 0.15 else 0; break
    acc+=e['token']
print(ok)" "$other"
}
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
# MTP_FORCE_DEPTH=3: every MTP round verifies 4 rows, so [5b] exercises the
# array-mask row split (S >= 3 at gqa 12 is MLX's unfused fallback).
# --max-concurrent 4: [8]-[10] batch plain slots; --prefix-cache-entries 0: the
# serial reruns those arms compare against must not restore (hybrid restore
# class 0.14-0.30 nats > the near-tie bar).
MLX_SERVE_MTP_FORCE_DEPTH=3 "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info --max-concurrent 4 --prefix-cache-entries 0 > "$LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for _ in $(seq 1 600); do curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && grep -q "ready" "$LOG" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$LOG"; exit 1; }; sleep 2; done
U="http://127.0.0.1:$PORT"
echo "[1] architecture + n-gram table"
arch=$(curl -s "$U/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['meta'].get('architecture',''))")
check "architecture" "$arch" "qwen4_exp"
check "ngram table log" "$(grep -c '\[qwen4\] n-gram table' "$LOG")" "1"
echo "[2] greedy short answer"
ans=$(curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}],"max_tokens":64,"temperature":0,"enable_thinking":false}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
check "mentions Paris" "$(echo "$ans" | grep -ci paris | sed 's/^0$/0/;s/^[1-9][0-9]*$/1/')" "1"
echo "[3] tool call + round trip"
body='{"messages":[{"role":"user","content":"What is the weather in Berlin right now? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":512,"temperature":0,"enable_thinking":false}'
tc=$(curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$body")
name=$(echo "$tc" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['choices'][0]['message'].get('tool_calls') or []; print(c[0]['function']['name'] if c else '')")
args_ok=$(echo "$tc" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['choices'][0]['message'].get('tool_calls') or []; a=json.loads(c[0]['function']['arguments']) if c else {}; print('1' if 'berlin' in json.dumps(a).lower() else '0')")
check "tool name" "$name" "get_weather"
check "args carry the city" "$args_ok" "1"
echo "[4] streaming deltas are clean"
st=$(curl -s -m 300 -N "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":48,"temperature":0,"stream":true,"enable_thinking":false}')
check "stream terminates" "$(echo "$st" | grep -c '^data: \[DONE\]')" "1"
check "no markup leak" "$(echo "$st" | grep -c '<tool_call>\|<think>\|<|im_')" "0"
echo "[5] past the QSA budget"
long=$(python3 -c "
import json
filler=' '.join(f'Fact {i}: the sky over city number {i} is blue.' for i in range(700))
print(json.dumps({'messages':[{'role':'user','content':filler+' The secret code is PELICAN-42. '+filler+' What is the secret code? Answer with the code only.'}],'max_tokens':32,'temperature':0,'enable_thinking':False}))")
la=$(curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$long" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'], '|', d['choices'][0]['message']['content'])")
echo "  prompt_tokens|answer: $la"
check "qsa engaged line" "$(grep -c '\[qsa\] sparse attention engaged' "$LOG")" "1"
check "needle recovered" "$(echo "$la" | grep -c 'PELICAN-42')" "1"
check "qsa prefill kernel engaged (mask arm of msv_attn_p256)" "$(grep -c '\[qsa-fused\] engaged' "$LOG")" "1"
check "qsa decode gather engaged (S=1 subset rows)" "$(grep -c '\[qsa-decode-gather\] engaged' "$LOG")" "1"
echo "[5b] MTP past the QSA budget (verify rows under the QSA mask)"
longm=$(echo "$long" | python3 -c "import sys,json; d=json.load(sys.stdin); d['enable_mtp']=True; print(json.dumps(d))")
lm=$(curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$longm" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
check "needle recovered under MTP" "$(echo "$lm" | grep -c 'PELICAN-42')" "1"
check "masked verify split engaged" "$(grep -c 'sdpa-split\] masked arm engaged' "$LOG")" "1"
echo "[6] MTP head: engagement + greedy equivalence"
base=$(curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Write a limerick about a cat."}],"max_tokens":80,"temperature":0,"enable_thinking":false,"enable_mtp":false}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
mtp=$(curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Write a limerick about a cat."}],"max_tokens":80,"temperature":0,"enable_thinking":false,"enable_mtp":true}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
check "mtp engaged" "$(grep -c 'spec-stats\] mode=mtp' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
if [ "${mtp:0:60}" = "${base:0:60}" ]; then check "mtp == serial (first 60 chars)" 1 1; else
  # Tie-aware acquittal (test_mtp_equivalence.sh): a verify width moves ties, a
  # plumbing bug diverges at a confident position. Serial top-2 gap at the first
  # divergent token must be <= 0.15 nats.
  gap=$(curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Write a limerick about a cat."}],"max_tokens":80,"temperature":0,"enable_thinking":false,"enable_mtp":false,"logprobs":true,"top_logprobs":2}' | python3 -c "
import sys,json; d=json.load(sys.stdin); mtp=sys.argv[1]; acc=''
for e in d['choices'][0]['logprobs']['content']:
    if not mtp.startswith(acc+e['token']):
        t=e['top_logprobs']; print(round(t[0]['logprob']-t[1]['logprob'],3) if len(t)>1 else 99); break
    acc+=e['token']
else: print(0)" "$mtp")
  echo "  mtp diverged; serial top-2 gap at the first divergent token: $gap nats"
  check "mtp == serial or near-tie divergence (gap <= 0.15)" "$(python3 -c "print(1 if float('${gap:-99}')<=0.15 else 0)")" "1"
fi
apr=$(grep 'spec-stats\] mode=mtp' "$LOG" | tail -1 | sed -E 's/.*avg_per_round=([0-9.]+).*/\1/')
echo "  avg accepted per round: $apr"
check "acceptance floor 0.5/round" "$(python3 -c "print(1 if float('${apr:-0}')>=0.5 else 0)")" "1"
echo "[7] image turn (vision tower + M-RoPE)"
IMAGE="$(dirname "$0")/fixtures/house.jpeg"
img=""
if [ -f "$IMAGE" ]; then
  B64=$(base64 -i "$IMAGE")
  img=$(python3 -c "
import json,sys
print(json.dumps({'messages':[{'role':'user','content':[{'type':'text','text':'What is the main subject of this image? One word.'},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+sys.argv[1]}}]}],'max_tokens':48,'temperature':0,'enable_thinking':False}))" "$B64")
fi
if [ -f "$MODEL/model-vision.safetensors" ] && [ -f "$IMAGE" ]; then
  resp=$(echo "$img" | curl -s -m 600 -w '\n%{http_code}' "$U/v1/chat/completions" -H 'content-type: application/json' -d @-)
  code=$(echo "$resp" | tail -1)
  ians=$(echo "$resp" | sed '$d' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null)
  echo "  -> $(echo "$ians" | tr '\n' ' ' | cut -c1-80)"
  check "image http 200" "$code" "200"
  check "image answer names the house" "$(echo "$ians" | grep -ciE 'house|home|building|cottage' | sed 's/^[1-9][0-9]*$/1/')" "1"
  check "vision encoder load line" "$(grep -c 'Vision encoder: Qwen3-VL ViT' "$LOG")" "1"
  check "m-rope engaged" "$(grep -c 'M-RoPE: 1 images' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
  echo "[7b] MTP on the image turn (head reads the slot's M-RoPE table)"
  nm7=$(grep -c 'spec-stats\] mode=mtp' "$LOG")
  imgm=$(echo "$img" | python3 -c "import sys,json; d=json.load(sys.stdin); d['enable_mtp']=True; print(json.dumps(d))")
  mans=$(echo "$imgm" | curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
  echo "  -> $(echo "$mans" | tr '\n' ' ' | cut -c1-80)"
  check "mtp engaged on the image turn" "$(python3 -c "print(1 if $(grep -c 'spec-stats\] mode=mtp' "$LOG") > $nm7 else 0)")" "1"
  check "mtp image answer == serial (tie-aware)" "$(same_or_tie "$img" "$mans")" "1"
  # Sampled AFTER an image turn: tower weights are lazy until first use.
  vis_bytes=$(curl -s "$U/props" | python3 -c "import sys,json; print(json.load(sys.stdin)['memory']['active_bytes'])")
else
  echo "  SKIP: pack has no model-vision.safetensors"
fi
echo "[8] two concurrent plain requests batch-decode (one forward for both slots)"
nb0=$(grep -c 'gdn batched decode engaged' "$LOG")
pa='{"messages":[{"role":"user","content":"Write a short story about a lighthouse keeper who finds a message in a bottle."}],"max_tokens":400,"temperature":0,"enable_thinking":false,"enable_mtp":false}'
pb='{"messages":[{"role":"user","content":"Explain how a bicycle stays upright while moving, step by step."}],"max_tokens":400,"temperature":0,"enable_thinking":false,"enable_mtp":false}'
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$pa" > "$LOG.8a" &
c1=$!
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$pb" > "$LOG.8b" &
c2=$!
wait $c1 $c2
ta=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$LOG.8a")
tb=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$LOG.8b")
echo "  A: $(echo "$ta" | tr '\n' ' ' | cut -c1-70)"
echo "  B: $(echo "$tb" | tr '\n' ' ' | cut -c1-70)"
check "batched decode engaged (slots=2)" "$(grep -c 'gdn batched decode engaged (slots=2)' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "A == serial (30 tokens, tie-aware)" "$(same_or_tie "$pa" "$ta")" "1"
check "B == serial (30 tokens, tie-aware)" "$(same_or_tie "$pb" "$tb")" "1"
echo "[9] two concurrent long prompts: batched decode under the QSA mask"
mk_long() { python3 -c "
import json,sys
code=sys.argv[1]; nonce=sys.argv[2]
filler=' '.join(f'Note {i}: the {nonce} river near town number {i} runs east.' for i in range(700))
print(json.dumps({'messages':[{'role':'user','content':filler+' The secret code is '+code+'. '+filler+' What is the secret code? Answer with the code only.'}],'max_tokens':24,'temperature':0,'enable_thinking':False,'enable_mtp':False}))" "$1" "$2"; }
la9=$(mk_long HERON-77 amber); lb9=$(mk_long OTTER-31 cobalt)
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$la9" > "$LOG.9a" &
c1=$!
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$lb9" > "$LOG.9b" &
c2=$!
wait $c1 $c2
ta=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$LOG.9a")
tb=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$LOG.9b")
echo "  A: $(echo "$ta" | tr '\n' ' ' | cut -c1-60)  B: $(echo "$tb" | tr '\n' ' ' | cut -c1-60)"
check "needle A" "$(echo "$ta" | grep -c 'HERON-77')" "1"
check "needle B" "$(echo "$tb" | grep -c 'OTTER-31')" "1"
check "batched decode engaged again past the budget" "$(python3 -c "print(1 if $(grep -c 'gdn batched decode engaged' "$LOG") > $nb0 else 0)")" "1"
check "A == serial (tie-aware)" "$(same_or_tie "$la9" "$ta")" "1"
check "B == serial (tie-aware)" "$(same_or_tie "$lb9" "$tb")" "1"
echo "[10] an MTP slot stays exclusive while a plain slot decodes beside it"
nm0=$(grep -c 'spec-stats\] mode=mtp' "$LOG")
pm='{"messages":[{"role":"user","content":"List ten European capitals with one fact each."}],"max_tokens":200,"temperature":0,"enable_thinking":false,"enable_mtp":true}'
pp='{"messages":[{"role":"user","content":"Describe the water cycle in five sentences."}],"max_tokens":200,"temperature":0,"enable_thinking":false,"enable_mtp":false}'
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$pm" > "$LOG.10m" &
c1=$!
sleep 0.5
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$pp" > "$LOG.10p" &
c2=$!
wait $c1 $c2
tm=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$LOG.10m")
tp=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$LOG.10p")
echo "  mtp: $(echo "$tm" | tr '\n' ' ' | cut -c1-60)  plain: $(echo "$tp" | tr '\n' ' ' | cut -c1-60)"
check "exactly one more mtp engagement" "$(python3 -c "print($(grep -c 'spec-stats\] mode=mtp' "$LOG") - $nm0)")" "1"
check "mtp answer == serial (tie-aware)" "$(same_or_tie "$pm" "$tm")" "1"
check "plain answer == serial (tie-aware)" "$(same_or_tie "$pp" "$tp")" "1"
echo "[11] --no-vision boot: tower absent, text works, media 400s by name"
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
LOG11="$LOG.novision"
sleep 20
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info --no-vision > "$LOG11" 2>&1 &
SPID=$!
for _ in $(seq 1 600); do curl -s "$U/health" >/dev/null 2>&1 && grep -q "Model ready" "$LOG11" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$LOG11"; exit 1; }; sleep 2; done
check "vision encoder load line absent" "$(grep -c 'Vision encoder: Qwen3-VL ViT' "$LOG11")" "0"
check "capabilities drop vision" "$(curl -s "$U/v1/models" | python3 -c "import sys,json; print(1 if 'vision' in json.load(sys.stdin)['data'][0].get('capabilities',[]) else 0)")" "0"
nv_bytes=$(curl -s "$U/props" | python3 -c "import sys,json; print(json.load(sys.stdin)['memory']['active_bytes'])")
if [ -n "${vis_bytes:-}" ]; then
  echo "  active_bytes with tower: $vis_bytes  without: $nv_bytes"
  check "resident memory lower without the tower (>= 400 MB)" "$(python3 -c "print(1 if $vis_bytes - $nv_bytes >= 400*1024*1024 else 0)")" "1"
fi
ans=$(curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}],"max_tokens":64,"temperature":0,"enable_thinking":false}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
check "text turn mentions Paris" "$(echo "$ans" | grep -ci paris | sed 's/^[1-9][0-9]*$/1/')" "1"
if [ -f "$IMAGE" ]; then
  resp=$(echo "$img" | curl -s -m 600 -w '\n%{http_code}' "$U/v1/chat/completions" -H 'content-type: application/json' -d @-)
  check "image turn http 400" "$(echo "$resp" | tail -1)" "400"
  check "400 names the missing tower" "$(echo "$resp" | grep -c 'vision tower')" "1"
fi
echo "passed $pass failed $fail"
[ "$fail" = 0 ]
