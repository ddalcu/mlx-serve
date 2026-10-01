#!/bin/bash
# test_mimo_ring_reuse.sh — MiMo's sliding layers keep a ring, and a reply
# longer than the ring compacts it past the prompt end. The next turn diverges
# there (the template re-renders the reply), so without a restore point every
# turn cold-prefilled ("clamp to N declined: SlidingRingRewindPastWindow").
# Pins, with thinking on (MiMo's default) and one hot entry:
#
#  1. An identical re-issue after a long reply restores from the prompt-end
#     ring checkpoint and its output is byte-identical to the cold run.
#  2. A turn that diverges right after the prompt (the previous reply
#     re-rendered) restores at the checkpoint, not cold, and its logprobs
#     match the same prompt prefilled cold within 0.5 nats up to any flip.
#  3. Both warm prefills are faster than the cold one.
#  4. A new session sharing only another's system prompt restores at the message start that
#     session's prefill marked, within the same 0.5-nat bar, and prefills faster than cold.
#
# Both cold arms start from a cache holding nothing they share: turn 1's whole prompt is a
# prefix of turn 2's rendering, so an unrelated raw completion evicts turn 2's entry first.
#
# Env: MLX_MODELS_DIR (default $HOME/.mlx-serve/models), PORT (default 19078), BINARY.

set -uo pipefail

MODEL="${MIMO_MODEL:-${MLX_MODELS_DIR:-$HOME/.mlx-serve/models}/MiMo-V2.6-Flash-MOPD}"
PORT="${PORT:-19078}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }
curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && { echo "fail: port $PORT is busy"; exit 1; }

LOG="$(mktemp)"
SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; rm -f "$LOG"; true' EXIT

"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --kv-quant 8 \
    --prefix-cache-entries 1 --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 900); do
    curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; exit 1; }
    sleep 1
done

BACKGROUND="$(python3 - <<'PY'
para = ("The kingdom of Avalon was beset by trials. Each season brought new "
        "challenges to its people, but the king remained steadfast. ")
print("Background:\n" + para * 30)
PY
)"
Q1="$BACKGROUND
Retell this story in about 800 words."
Q2="Now retell it again from the point of view of the king, in about 100 words."
REPLY="The kingdom endured; its king held firm through every season."

# Every request is greedy and stores its whole response for the byte comparison.
ask() {
    curl -sf --max-time 1800 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$1"
}
turn1() { jq -nc --arg q "$Q1" '{messages:[{role:"user",content:$q}],max_tokens:2400,temperature:0,stream:false}'; }
turn2() {
    jq -nc --arg q "$Q1" --arg r "$REPLY" --arg q2 "$Q2" \
        '{messages:[{role:"user",content:$q},{role:"assistant",content:$r},{role:"user",content:$q2}],max_tokens:300,temperature:0,stream:false,logprobs:true,top_logprobs:1}'
}
# A raw completion shares no token with the chat template, so its entry restores nothing below.
unrelated() { jq -nc '{prompt:"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor.",max_tokens:8,temperature:0,stream:false}'; }
text() { echo "$1" | jq -r '(.choices[0].message.reasoning_content // "") + "\u0001" + (.choices[0].message.content // "")'; }
field() { echo "$1" | jq -r "$2"; }
ring_hits() { grep -c '\[hot-cache\] ring checkpoint @' "$LOG"; }

EC=0
fail() { echo "FAIL: $*"; EC=1; }

# Turn 2 first, while nothing is cached: its cold answer is the reference.
T2_COLD=$(ask "$(turn2)") || { echo "fail: cold turn 2"; tail -20 "$LOG"; exit 1; }
# With one entry this commit evicts the turn-2 entry, so turn 1 starts cold too.
curl -sf --max-time 600 -X POST "$BASE/v1/completions" -H 'Content-Type: application/json' -d "$(unrelated)" >/dev/null \
    || { echo "fail: unrelated completion"; tail -20 "$LOG"; exit 1; }
T1_COLD=$(ask "$(turn1)") || { echo "fail: cold turn 1"; tail -20 "$LOG"; exit 1; }
GEN=$(field "$T1_COLD" '.usage.completion_tokens')
if [ "$GEN" -lt 600 ]; then
    # The ring keeps 384 to 640 rows; a reply this short may leave the prompt end inside it.
    echo "SKIP: turn 1 generated only $GEN tokens; the ring may still hold the prompt end"
    exit 0
fi
HITS0=$(ring_hits)
T1_WARM=$(ask "$(turn1)") || { echo "fail: warm turn 1"; tail -20 "$LOG"; exit 1; }
HITS1=$(ring_hits)
T2_WARM=$(ask "$(turn2)") || { echo "fail: warm turn 2"; tail -20 "$LOG"; exit 1; }
HITS2=$(ring_hits)

echo "turn1 cold: cached_n=$(field "$T1_COLD" .timings.cached_n) prompt_ms=$(field "$T1_COLD" .timings.prompt_ms) generated=$GEN"
echo "turn1 warm: cached_n=$(field "$T1_WARM" .timings.cached_n) prompt_ms=$(field "$T1_WARM" .timings.prompt_ms)"
echo "turn2 cold: cached_n=$(field "$T2_COLD" .timings.cached_n) prompt_ms=$(field "$T2_COLD" .timings.prompt_ms)"
echo "turn2 warm: cached_n=$(field "$T2_WARM" .timings.cached_n) prompt_ms=$(field "$T2_WARM" .timings.prompt_ms)"

[ "$(field "$T2_COLD" .timings.cached_n)" = 0 ] || fail "cold turn 2 restored a prefix"
[ "$(field "$T1_COLD" .timings.cached_n)" = 0 ] || fail "cold turn 1 restored a prefix"
[ "$(text "$T1_COLD")" = "$(text "$T1_WARM")" ] || fail "identical re-issue diverged from its cold run"
[ "$HITS1" -gt "$HITS0" ] || fail "identical re-issue did not restore from the ring checkpoint"
[ "$(field "$T1_WARM" .timings.cached_n)" -gt 0 ] || fail "identical re-issue cold-prefilled"
# A divergent restore reuses rows another forward shape computed, so greedy may flip at a
# near-tie: a cold prefill at another chunk width moves this prompt's logprobs by tenths of
# a nat and flips it just as early. Rows restored from the wrong place move them by nats.
DRIFT=$(python3 - "$T2_COLD" "$T2_WARM" <<'PY2'
import json, sys
a, b = (json.loads(x)["choices"][0]["logprobs"]["content"] for x in sys.argv[1:3])
n = 0
worst = 0.0
while n < min(len(a), len(b)) and a[n]["token"] == b[n]["token"]:
    worst = max(worst, abs(a[n]["logprob"] - b[n]["logprob"]))
    n += 1
print(f"{n} {len(a)} {worst:.4f}")
PY2
)
read -r AGREE TOTAL WORST <<< "$DRIFT"
echo "turn2 cold vs warm: tokens agree for $AGREE of $TOTAL; max |dlogprob| over them $WORST nats"
python3 -c "import sys; sys.exit(0 if float('$WORST') <= 0.5 else 1)" || fail "restored turn 2 logprobs drift past 0.5 nats"
[ "$HITS2" -gt "$HITS1" ] || fail "turn 2 did not restore from the ring checkpoint"
[ "$(field "$T2_WARM" .timings.cached_n)" -gt 0 ] || fail "turn 2 cold-prefilled"
python3 -c "import sys; sys.exit(0 if $(field "$T1_WARM" .timings.prompt_ms) < 0.7 * $(field "$T1_COLD" .timings.prompt_ms) else 1)" \
    || fail "warm turn 1 prefill not under 0.7x cold"
python3 -c "import sys; sys.exit(0 if $(field "$T2_WARM" .timings.prompt_ms) < 0.7 * $(field "$T2_COLD" .timings.prompt_ms) else 1)" \
    || fail "warm turn 2 prefill not under 0.7x cold"

# 4. A new session that shares only another session's system prompt (an agent's subagent, a
#    second chat) restores at the message start the first session's prefill marked, though that
#    session's checkpoints and ring all lie past where the two diverge.
SYSTEM="$(python3 - <<'PY'
import random
r = random.Random(7)
w = "agent tool file cache ring window layer prompt reply token session restore commit budget chunk".split()
print("You are a careful agent.\n" + "\n".join(f"Rule {i}: " + " ".join(r.choice(w) for _ in range(30)) for i in range(60)))
PY
)"
task() { python3 -c "import random,sys; r=random.Random($1); w='trace the ring path and fix the bill then pin it with a test'.split(); print('Task: '+' '.join(r.choice(w) for _ in range(500)))"; }
session() {
    jq -nc --arg s "$SYSTEM" --arg q "$(task "$1")" \
        '{messages:[{role:"system",content:$s},{role:"user",content:$q}],max_tokens:48,temperature:0,stream:false,logprobs:true,top_logprobs:1,chat_template_kwargs:{enable_thinking:false}}'
}
S_B_COLD=$(ask "$(session 2)") || { echo "fail: cold session B"; tail -20 "$LOG"; exit 1; }
curl -sf --max-time 600 -X POST "$BASE/v1/completions" -H 'Content-Type: application/json' -d "$(unrelated)" >/dev/null \
    || { echo "fail: unrelated completion"; tail -20 "$LOG"; exit 1; }
S_A=$(ask "$(session 1)") || { echo "fail: session A"; tail -20 "$LOG"; exit 1; }
HITS3=$(ring_hits)
S_B_WARM=$(ask "$(session 2)") || { echo "fail: warm session B"; tail -20 "$LOG"; exit 1; }
HITS4=$(ring_hits)
echo "session A: cached_n=$(field "$S_A" .timings.cached_n) prompt_n=$(field "$S_A" .timings.prompt_n)"
echo "session B cold: cached_n=$(field "$S_B_COLD" .timings.cached_n) prompt_ms=$(field "$S_B_COLD" .timings.prompt_ms)"
echo "session B warm: cached_n=$(field "$S_B_WARM" .timings.cached_n) prompt_ms=$(field "$S_B_WARM" .timings.prompt_ms)"
[ "$(field "$S_A" .timings.cached_n)" = 0 ] || fail "session A restored a prefix"
[ "$HITS4" -gt "$HITS3" ] || fail "session B did not restore from a ring checkpoint"
# The marks sit at message starts: the restore covers the whole system message.
SYS_TOKENS=$(python3 -c "print(int($(field "$S_B_COLD" .timings.prompt_n) * 0.6))")
[ "$(field "$S_B_WARM" .timings.cached_n)" -gt "$SYS_TOKENS" ] || fail "session B restored less than its shared system prompt"
DRIFT=$(python3 - "$S_B_COLD" "$S_B_WARM" <<'PY2'
import json, sys
a, b = (json.loads(x)["choices"][0]["logprobs"]["content"] for x in sys.argv[1:3])
n = 0
worst = 0.0
while n < min(len(a), len(b)) and a[n]["token"] == b[n]["token"]:
    worst = max(worst, abs(a[n]["logprob"] - b[n]["logprob"]))
    n += 1
print(f"{n} {len(a)} {worst:.4f}")
PY2
)
read -r AGREE TOTAL WORST <<< "$DRIFT"
echo "session B cold vs warm: tokens agree for $AGREE of $TOTAL; max |dlogprob| over them $WORST nats"
[ "$TOTAL" -gt 0 ] || fail "session B returned no logprobs to compare"
python3 -c "import sys; sys.exit(0 if float('$WORST') <= 0.5 else 1)" || fail "session B's restored logprobs drift past 0.5 nats"
python3 -c "import sys; sys.exit(0 if $(field "$S_B_WARM" .timings.prompt_ms) < 0.7 * $(field "$S_B_COLD" .timings.prompt_ms) else 1)" \
    || fail "warm session B prefill not under 0.7x cold"
grep '\[hot-cache\]' "$LOG" | head -30

[ $EC = 0 ] && echo "PASS"
exit $EC
