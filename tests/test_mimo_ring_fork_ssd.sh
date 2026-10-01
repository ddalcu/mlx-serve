#!/bin/bash
# test_mimo_ring_fork_ssd.sh — MiMo's sliding layers keep a ring, so an entry restores only at its
# end or at a ring checkpoint. A client that sends a side request per turn (the whole conversation
# plus an appended <system-reminder>) used to leave the next main turn nothing to restore once the
# count cap evicted the main entry: it matched the side entry only up to the reminder, below every
# row that entry's ring kept, and cold-prefilled the conversation every turn. Pins, thinking on:
#
#  1. main turn, side request, an unrelated request evicting the main entry (two hot entries):
#     the next main turn restores from the side entry in RAM. The unrelated request's own lookup
#     declines at the ring clamp and must not promote the entry it declined on.
#  2. The SSD tier persists ringed entries as chunks plus ring files (manifest v9).
#  3. After a restart the same main turn restores from the SSD tier, and its logprobs match the
#     RAM-restored run within 0.5 nats up to any flip; a turn diverging at the reminder then restores
#     too (from either tier).
#  4. Both main turn 2 restores land past the reply: at the message start the side request's
#     prefill marked (a ring file at that position after the restart), not at turn 1's prompt end.
#
# Env: MLX_MODELS_DIR (default $HOME/.mlx-serve/models), MIMO_MODEL, PORT (default 18931), BINARY.

set -uo pipefail

MODEL="${MIMO_MODEL:-${MLX_MODELS_DIR:-$HOME/.mlx-serve/models}/MiMo-V2.6-Flash-MOPD}"
PORT="${PORT:-18931}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build mlx-serve first"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }
curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && { echo "fail: port $PORT is busy"; exit 1; }

# The SSD tier lives under $HOME/.mlx-serve/kv-cache: an isolated cache directory keeps the user's untouched.
MIMO_TEST_CACHE_DIR="$(mktemp -d)"
LOG="$(mktemp)"
SERVER_PID=""
stop() {
    [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
    SERVER_PID=""
    true
}
trap 'stop; rm -rf "$MIMO_TEST_CACHE_DIR" "$LOG"' EXIT

boot() {
    : > "$LOG"
    MLX_SERVE_KV_CACHE_DIR="$MIMO_TEST_CACHE_DIR" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --kv-quant 8 \
        --prefix-cache-entries 2 --prefix-cache-disk 16GB --log-level info > "$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 900); do
        curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; exit 1; }
        sleep 1
    done
    echo "fail: server never came up"; exit 1
}

BACKGROUND="$(python3 - <<'PY'
para = ("The kingdom of Avalon was beset by trials. Each season brought new "
        "challenges to its people, but the king remained steadfast. ")
print("Background:\n" + para * 30)
PY
)"
# Longer than the ring keeps (window + slack), so the side entry's own ring never reaches the fork.
REMINDER="$(python3 - <<'PY'
rule = ("Keep the dashboard line under twelve words, name the current task, "
        "and never repeat an earlier line verbatim. ")
print("<system-reminder>\n" + rule * 40 + "Reply with the dashboard line only.\n</system-reminder>")
PY
)"
Q1="$BACKGROUND
Retell this story in about 800 words."
Q2="Now retell it again from the point of view of the king, in about 100 words."
Q3="Which season was the hardest for the people? Answer in two sentences."
TITLE="Write a five-word title for a poem about the sea."

ask() {
    curl -sf --max-time 1800 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$1"
}
main1() { jq -nc --arg q "$Q1" '{messages:[{role:"user",content:$q}],max_tokens:2400,temperature:0,stream:false}'; }
# The conversation so far (the main reply as content only, as a client sends it back) plus one user turn.
after() {
    jq -nc --arg q "$Q1" --arg r "$REPLY" --arg u "$1" --argjson n "$2" \
        '{messages:[{role:"user",content:$q},{role:"assistant",content:$r},{role:"user",content:$u}],max_tokens:$n,temperature:0,stream:false,logprobs:true,top_logprobs:1}'
}
title() { jq -nc --arg q "$TITLE" '{messages:[{role:"user",content:$q}],max_tokens:32,temperature:0,stream:false}'; }
field() { echo "$1" | jq -r "$2"; }
count() { grep -c "$1" "$LOG"; }

EC=0
fail() { echo "FAIL: $*"; EC=1; }

# ── Boot 1: the client's pattern in RAM ──
boot
M1=$(ask "$(main1)") || { echo "fail: main turn 1"; tail -20 "$LOG"; exit 1; }
GEN=$(field "$M1" '.usage.completion_tokens')
if [ "$GEN" -lt 600 ]; then
    # The ring keeps 384 to 640 rows; a reply this short may leave the prompt end inside it.
    echo "SKIP: main turn 1 generated only $GEN tokens; the ring may still hold the prompt end"
    exit 0
fi
REPLY=$(field "$M1" '.choices[0].message.content')

RING0=$(count '\[hot-cache\] ring checkpoint @')
ask "$(after "$REMINDER" 48)" > /dev/null || { echo "fail: side request"; tail -20 "$LOG"; exit 1; }
RING1=$(count '\[hot-cache\] ring checkpoint @')
[ "$RING1" -gt "$RING0" ] || fail "the side request did not restore from the main entry's ring checkpoint"
grep -q '\[hot-cache\] inherited ring checkpoints' "$LOG" || fail "the side entry inherited no ring checkpoint"

EVICT0=$(count 'evicted LRU entry (count cap')
ask "$(title)" > /dev/null || { echo "fail: unrelated request"; tail -20 "$LOG"; exit 1; }
[ "$(count 'evicted LRU entry (count cap')" -gt "$EVICT0" ] || fail "the unrelated request evicted nothing"

DECLINE0=$(count 'declined: SlidingRingRewindPastWindow')
M2=$(ask "$(after "$Q2" 300)") || { echo "fail: main turn 2"; tail -20 "$LOG"; exit 1; }
RING2=$(count '\[hot-cache\] ring checkpoint @')
[ "$RING2" -gt "$RING1" ] || fail "main turn 2 did not restore from the side entry in RAM"
[ "$(count 'declined: SlidingRingRewindPastWindow')" = "$DECLINE0" ] || fail "main turn 2 declined its clamp"
[ "$(field "$M2" .timings.cached_n)" -gt 0 ] || fail "main turn 2 cold-prefilled"
P1=$(field "$M1" .usage.prompt_tokens)
[ "$(field "$M2" .timings.cached_n)" -gt "$P1" ] || fail "main turn 2 restored at or below turn 1's prompt end, not at the side request's mark"
echo "main turn 2 (RAM): cached_n=$(field "$M2" .timings.cached_n) prompt_ms=$(field "$M2" .timings.prompt_ms) of $(field "$M2" .usage.prompt_tokens) prompt tokens"

# The flush runs after the response: wait for the three long entries (the title request is under
# the persist floor) to land before the restart.
for _ in $(seq 1 120); do
    [ "$(count '\[disk-cache\] persisted')" -ge 3 ] && break
    sleep 1
done
sleep 2
echo "boot 1 persisted $(count '\[disk-cache\] persisted') entries"
grep '\[hot-cache\]\|\[disk-cache\]' "$LOG" | head -40
stop

RINGS=$(ls "$MIMO_TEST_CACHE_DIR"/*/e*/r*.safetensors 2>/dev/null | wc -l | tr -d ' ')
[ "$RINGS" -gt 0 ] || fail "no ring file on disk"
grep -lq '"v":9' "$MIMO_TEST_CACHE_DIR"/*/e*/meta.json 2>/dev/null || fail "no v9 manifest on disk"

# ── Boot 2: a restart restores from the SSD tier ──
boot
DISK0=$(count '\[disk-cache\] restored .*ring@')
M2D=$(ask "$(after "$Q2" 300)") || { echo "fail: main turn 2 after restart"; tail -20 "$LOG"; exit 1; }
[ "$(count '\[disk-cache\] restored .*ring@')" -gt "$DISK0" ] || fail "main turn 2 after restart did not restore from SSD"
[ "$(field "$M2D" .timings.cached_n)" -gt 0 ] || fail "main turn 2 after restart cold-prefilled"
[ "$(field "$M2D" .timings.cached_n)" -gt "$P1" ] || fail "main turn 2 after restart restored at or below turn 1's prompt end"
echo "main turn 2 (SSD): cached_n=$(field "$M2D" .timings.cached_n) prompt_ms=$(field "$M2D" .timings.prompt_ms)"
DRIFT=$(python3 - "$M2" "$M2D" <<'PY'
import json, sys
a, b = (json.loads(x)["choices"][0]["logprobs"]["content"] for x in sys.argv[1:3])
n = 0
worst = 0.0
while n < min(len(a), len(b)) and a[n]["token"] == b[n]["token"]:
    worst = max(worst, abs(a[n]["logprob"] - b[n]["logprob"]))
    n += 1
print(f"{n} {len(a)} {worst:.4f}")
PY
)
read -r AGREE TOTAL WORST <<< "$DRIFT"
echo "main turn 2 RAM vs SSD restore: tokens agree for $AGREE of $TOTAL; max |dlogprob| over them $WORST nats"
python3 -c "import sys; sys.exit(0 if float('$WORST') <= 0.5 else 1)" || fail "SSD-restored logprobs drift past 0.5 nats"

DISK1=$(count '\[disk-cache\] restored .*ring@')
M3=$(ask "$(after "$Q3" 64)") || { echo "fail: main turn 3"; tail -20 "$LOG"; exit 1; }
[ "$(count '\[disk-cache\] restored .*ring@')" -gt "$DISK1" ] || [ "$(field "$M3" .timings.cached_n)" -gt 0 ] \
    || fail "a turn diverging at the reminder restored nothing after the restart"
echo "main turn 3: cached_n=$(field "$M3" .timings.cached_n)"
grep '\[hot-cache\]\|\[disk-cache\]' "$LOG" | head -20
stop

[ $EC = 0 ] && echo "PASS"
exit $EC
