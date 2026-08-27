#!/usr/bin/env bash
# Qwen3.8-Flash-Next (qwen4_exp) live end-to-end on the converted pack.
# Boots the pack, then: architecture advertised, a short greedy answer, a tool
# call + round-trip, streaming delta cleanliness, and a prompt past the QSA
# budget (2048 tokens) — asserted through the server's own
# `[qsa] sparse attention engaged` line, since a dense fallback answers
# plausibly too. [7] sends an image (tower + M-RoPE engagement lines) and
# SKIPs when the pack ships no `model-vision.safetensors`. SKIPs without the pack.
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
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
# MTP_FORCE_DEPTH=3: every MTP round verifies 4 rows, so [5b] exercises the
# array-mask row split (S >= 3 at gqa 12 is MLX's unfused fallback).
MLX_SERVE_MTP_FORCE_DEPTH=3 "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info > "$LOG" 2>&1 &
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
if [ -f "$MODEL/model-vision.safetensors" ] && [ -f "$IMAGE" ]; then
  B64=$(base64 -i "$IMAGE")
  img=$(python3 -c "
import json,sys
print(json.dumps({'messages':[{'role':'user','content':[{'type':'text','text':'What is the main subject of this image? One word.'},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+sys.argv[1]}}]}],'max_tokens':48,'temperature':0,'enable_thinking':False}))" "$B64")
  resp=$(echo "$img" | curl -s -m 600 -w '\n%{http_code}' "$U/v1/chat/completions" -H 'content-type: application/json' -d @-)
  code=$(echo "$resp" | tail -1)
  ians=$(echo "$resp" | sed '$d' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null)
  echo "  -> $(echo "$ians" | tr '\n' ' ' | cut -c1-80)"
  check "image http 200" "$code" "200"
  check "image answer names the house" "$(echo "$ians" | grep -ciE 'house|home|building|cottage' | sed 's/^[1-9][0-9]*$/1/')" "1"
  check "vision encoder load line" "$(grep -c 'Vision encoder: Qwen3-VL ViT' "$LOG")" "1"
  check "m-rope engaged" "$(grep -c 'M-RoPE: 1 images' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
else
  echo "  SKIP: pack has no model-vision.safetensors"
fi
echo "passed $pass failed $fail"
[ "$fail" = 0 ]
