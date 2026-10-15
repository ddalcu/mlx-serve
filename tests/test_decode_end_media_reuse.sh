#!/bin/bash
# An image conversation's next turn resumes where the previous reply ended, and answers as a cold prefill does.
#
# A hybrid restore lands only on an SSM checkpoint. A text turn snapshots where decode stopped
# (`appendDecodeEndCheckpoint`), so the next tool round resumes after its own reply; an image turn
# was excluded, so every later turn of an image conversation re-read its previous reply. Checks, on a
# hybrid vision model:
#   1. the image turn logs a decode-end checkpoint;
#   2. the next turn (turn 1 + its reply + a new question) restores at least turn 1's prompt + reply;
#   3. its greedy answer equals a cold server's, or first differs where the reference itself is close (<= 0.5 nats, the hybrid chunking class): a
#      wrong position offset diverges at a confident token.
#
# Env: MEDIA_REUSE_MODEL (default Qwen3.5-0.8B-MLX-4bit), PORT (default 19091), BINARY.
set -uo pipefail
cd "$(dirname "$0")/.."
source tests/_lib_models.sh
MODEL="${MEDIA_REUSE_MODEL:-$(find_model mlx-community/Qwen3.5-0.8B-MLX-4bit)}"
PORT="${PORT:-19091}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
B="http://127.0.0.1:$PORT"
[ -d "$MODEL" ] || { echo "SKIP: no hybrid vision model ($MODEL)"; exit 0; }
[ -x "$BIN" ] || { echo "FAIL: build mlx-serve first"; exit 1; }

WORK=$(mktemp -d)
PID=""
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null; [ -n "${KEEP_WORK:-}" ] || rm -rf "$WORK"' EXIT
FAIL=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; FAIL=1; }

boot() { # $1 log, $2.. extra flags
    local log="$1"; shift
    HOME="$WORK/home" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level debug \
        --no-mtp --no-pld --no-drafter "$@" > "$log" 2>&1 &
    PID=$!
    for _ in $(seq 240); do curl -sf "$B/health" >/dev/null 2>&1 && return 0; kill -0 $PID 2>/dev/null || return 1; sleep 0.5; done
    return 1
}
stop() { kill $PID 2>/dev/null; wait $PID 2>/dev/null; PID=""; }

python3 - "$WORK" <<'PY'
import base64, json, sys
w = sys.argv[1]
img = "data:image/jpeg;base64," + base64.b64encode(open("tests/fixtures/house.jpeg", "rb").read()).decode()
q1 = [{"type": "text", "text": "Describe this picture in detail: the building, its colours, the sky and anything else you notice."},
      {"type": "image_url", "image_url": {"url": img}}]
json.dump({"messages": [{"role": "user", "content": q1}], "max_tokens": 1500, "temperature": 0}, open(f"{w}/turn1.json", "w"))
json.dump(q1, open(f"{w}/q1.json", "w"))
PY

mkdir -p "$WORK/home"
boot "$WORK/warm.log" --prefix-cache-entries 4 || { fail "warm server boots"; exit 1; }
curl -s "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @"$WORK/turn1.json" > "$WORK/r1.json"
python3 - "$WORK" <<'PY'
import json, sys
w = sys.argv[1]
r1 = json.load(open(f"{w}/r1.json"))
m = r1["choices"][0]["message"]
q1 = json.load(open(f"{w}/q1.json"))
# The reply goes back as agents send it, reasoning included, so the turn re-renders token for token.
reply = {"role": "assistant", "content": m.get("content") or ""}
if m.get("reasoning_content"): reply["reasoning_content"] = m["reasoning_content"]
msgs = [{"role": "user", "content": q1}, reply,
        {"role": "user", "content": "Now in one sentence: what season does it look like, and why?"}]
json.dump({"messages": msgs, "max_tokens": 40, "temperature": 0, "logprobs": True, "top_logprobs": 2}, open(f"{w}/turn2.json", "w"))
open(f"{w}/turn1_total", "w").write(str(r1["usage"]["prompt_tokens"] + r1["usage"]["completion_tokens"]))
PY
/usr/bin/grep -aq "decode-end checkpoint at" "$WORK/warm.log" && pass "the image turn takes a decode-end checkpoint" \
    || fail "the image turn takes a decode-end checkpoint ($(/usr/bin/grep -ao 'decode-end checkpoint skipped[^$]*' "$WORK/warm.log" | head -1))"
curl -s "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @"$WORK/turn2.json" > "$WORK/warm2.json"
reused=$(/usr/bin/grep -aoE "reused [0-9]+/[0-9]+ tokens" "$WORK/warm.log" | tail -1 | awk '{print $2}' | cut -d/ -f1)
t1=$(cat "$WORK/turn1_total")
[ "${reused:-0}" -ge $((t1 - 2)) ] && pass "turn 2 restores past turn 1's reply ($reused >= $t1 - 2)" \
    || fail "turn 2 restores past turn 1's reply (reused ${reused:-0}, turn 1 was $t1)"
stop

boot "$WORK/cold.log" --prefix-cache-entries 0 || { fail "cold server boots"; exit 1; }
curl -s "$B/v1/chat/completions" -H 'Content-Type: application/json' -d @"$WORK/turn2.json" > "$WORK/cold2.json"
stop

python3 - "$WORK" <<'PY' && pass "the warm answer equals the cold one (or first differs at a near-tie)" || fail "the warm answer equals the cold one"
import json, sys
w = sys.argv[1]
warm, cold = (json.load(open(f"{w}/{n}2.json")) for n in ("warm", "cold"))
wt = [e["token"] for e in warm["choices"][0]["logprobs"]["content"]]
ct = cold["choices"][0]["logprobs"]["content"]
for i, (a, e) in enumerate(zip(wt, ct)):
    if a != e["token"]:
        top = e["top_logprobs"]
        gap = top[0]["logprob"] - top[1]["logprob"] if len(top) > 1 else 99
        print(f"  first difference at token {i}: warm {a!r} cold {e['token']!r}, cold top-2 gap {gap:.3f} nats")
        sys.exit(0 if gap <= 0.5 else 1)
print(f"  identical over {min(len(wt), len(ct))} tokens")
PY
exit $FAIL
