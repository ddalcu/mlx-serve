#!/usr/bin/env bash
# Concurrent MTP users on one model: every stream's greedy output at a FIXED draft depth
# must be byte-identical to its solo run (the byte bar for spec decode), whether the
# rounds ride one batched verify (qwen3_5 dense/MoE) or interleave with their own head
# state (qwen4_exp). A crowd of 4 must still complete on the plain batched tick.
# Usage: MTP_BATCHED_MODEL=<dir with an MTP head> ./tests/test_mtp_batched.sh [port]
set -u
MODEL="${MTP_BATCHED_MODEL:?set MTP_BATCHED_MODEL}"
PORT=${1:-18850}; BASE="http://127.0.0.1:$PORT"
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
LOG=$(mktemp); OUT=$(mktemp -d)
MLX_SERVE_MTP_FORCE_DEPTH=2 MLX_SERVE_ROUND_COST_PERSIST=0 "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --mtp --max-concurrent 8 --prefix-cache-entries 0 > "$LOG" 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null; rm -rf "$LOG" "$OUT"' EXIT
for i in $(seq 1 240); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "$BASE/health" >/dev/null || { echo -e "${RED}FAIL${NC} server did not start"; tail -20 "$LOG"; exit 1; }
IS_QWEN4=$(python3 -c "import json,sys; print(1 if json.load(open(sys.argv[1]+'/config.json')).get('model_type')=='qwen4_exp' else 0)" "$MODEL")

P=("Write a short essay about the history of quantum computing." "Explain how a compiler turns source code into machine code." "Describe the water cycle for a graduate seminar." "Summarize the causes of the French revolution.")
req() { python3 -c "import json,sys; print(json.dumps({'model':'mlx-serve','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':150,'temperature':0}))" "$1" |
    curl -s -m 900 -X POST -H 'Content-Type: application/json' -d @- "$BASE/v1/chat/completions" |
    python3 -c "import sys,json; j=json.load(sys.stdin); print(j['choices'][0]['message']['content'])"; }

# qwen4 rounds stay solo and two of them interleave; three go plain (a B=3 tick).
N=$([ "$IS_QWEN4" = 1 ] && echo 2 || echo 3); LAST=$((N-1))
req "${P[0]}" > /dev/null
for i in $(seq 0 $LAST); do req "${P[$i]}" > "$OUT/solo$i"; done
pids=(); for i in $(seq 0 $LAST); do req "${P[$i]}" > "$OUT/conc$i" & pids+=($!); done; wait "${pids[@]}"
# A batched verify is a B>1 forward: like the plain batched tick it is not bit-identical
# to serial, so a divergence is acquitted at a serial top-2 gap <= 0.15 nats (the MTP
# equivalence bar). qwen4 rounds are solo forwards and must match byte for byte.
fail=0
for i in $(seq 0 $LAST); do
    if cmp -s "$OUT/solo$i" "$OUT/conc$i"; then continue; fi
    if [ "$IS_QWEN4" = 1 ]; then
        echo -e "${RED}FAIL${NC} stream $i at N=$N differs from its solo run"; diff "$OUT/solo$i" "$OUT/conc$i" | head -6; fail=1; continue
    fi
    gap=$(python3 - "$BASE" "${P[$i]}" "$OUT/solo$i" "$OUT/conc$i" <<'PYEOF'
import sys, json, urllib.request
base, prompt, solo, conc = sys.argv[1], sys.argv[2], open(sys.argv[3]).read(), open(sys.argv[4]).read()
def tok(t):
    r = urllib.request.urlopen(urllib.request.Request(base + "/tokenize", json.dumps({"content": t}).encode(), {"Content-Type": "application/json"}))
    return json.load(r)["tokens"]
a, b = tok(solo.rstrip("\n")), tok(conc.rstrip("\n"))
idx = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
body = {"model": "mlx-serve", "messages": [{"role": "user", "content": prompt}], "max_tokens": 150, "temperature": 0, "logprobs": True, "top_logprobs": 2, "enable_mtp": False}
r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}))
c = json.load(r)["choices"][0]["logprobs"]["content"]
t = c[min(idx, len(c) - 1)]["top_logprobs"]
print(round(t[0]["logprob"] - t[1]["logprob"], 4), idx)
PYEOF
)
    g=${gap%% *}; idx=${gap##* }
    if python3 -c "import sys; sys.exit(0 if float('$g') <= 0.15 else 1)"; then
        echo -e "  ${YELLOW}near-tie${NC} stream $i diverged at token $idx, serial top-2 gap $g nats: acquitted"
    else
        echo -e "${RED}FAIL${NC} stream $i at N=$N diverged at token $idx with a serial top-2 gap of $g nats"; diff "$OUT/solo$i" "$OUT/conc$i" | head -6; fail=1
    fi
done
[ "$fail" = 0 ] && echo -e "${GREEN}PASS${NC} $N concurrent MTP streams match solo (fixed depth 2; near-ties acquitted on the batched verify)"
if [ "$IS_QWEN4" = 1 ]; then
    grep -q "gdn batched verify engaged" "$LOG" && { echo -e "${RED}FAIL${NC} qwen4 rounds must stay solo (no batched verify yet)"; fail=1; }
    [ "$fail" = 0 ] && echo -e "${GREEN}PASS${NC} qwen4 MTP users interleave on their own head state"
else
    grep -q "gdn batched verify engaged" "$LOG" || { echo -e "${RED}FAIL${NC} no batched verify engaged"; fail=1; }
    [ "$fail" = 0 ] && echo -e "${GREEN}PASS${NC} batched verify engaged: $(grep -o 'gdn batched verify engaged.*' "$LOG" | head -1)"
fi

pids=(); for i in 0 1 2 3; do req "${P[$i]}" > "$OUT/crowd$i" & pids+=($!); done; wait "${pids[@]}"
for i in 0 1 2 3; do [ -s "$OUT/crowd$i" ] || { echo -e "${RED}FAIL${NC} crowd stream $i returned nothing"; fail=1; }; done
if [ "$IS_QWEN4" = 0 ]; then
    grep -q "gdn batched decode engaged (slots=4)" "$LOG" || { echo -e "${YELLOW}NOT RUN${NC} the four MTP streams never overlapped into one plain batched tick"; }
fi
[ "$fail" = 0 ] && echo -e "${GREEN}PASS${NC} four MTP streams complete (crowd arm)"
grep -ciE "panic|Segmentation" "$LOG" | grep -q "^0$" || { echo -e "${RED}FAIL${NC} server log has a crash"; fail=1; }
exit $fail
