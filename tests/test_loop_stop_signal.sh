#!/bin/bash
# test_loop_stop_signal.sh — the degenerate-tail guard's WIRE contract on every surface that can emit it: a loop cut
# reports `finish_reason: "stop"` (never "length": agents compacted on it) with `finish_details:{"type":
# "repetition_loop"}` beside it, on chat and completions, stream and non-stream; tool-call parsing is suppressed for
# that cause. [1/5] a non-streaming body is TRIMMED to the span start (an agent resends the cut turn and resumes its
# own loop); with thinking on the same holds for reasoning_content. Streaming keeps the tail (a delta cannot be
# retracted).
#
#   LOOP_TEST_MODEL=<small chat model> ./tests/test_loop_stop_signal.sh [port]   # SKIPs without one
#
# Hermetic twins: degenerateTail, loopStopDecision, shouldParseToolCalls, finishDetailsField, loopTrimmedIds.

set -u
source "$(dirname "$0")/lib/server.sh"

PORT="${1:-11266}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${LOOP_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

[ -x "$BINARY" ] || { echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"; exit 1; }
[ -d "$MODEL" ] || { echo "[skip] model not found: $MODEL (set LOOP_TEST_MODEL)"; exit 0; }

LOG="$(mktemp)"
cleanup() {
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    rm -f "$LOG"
}
trap cleanup EXIT

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
sleep 0.5
"$BINARY" --model "$MODEL" --serve --port "$PORT" --log-file off --log-level info > "$LOG" 2>&1 &
SRV=$!
srv_wait_health "http://127.0.0.1:$PORT" 240 1 "$SRV" || true
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
    echo "[fail] server did not come up"; tail -20 "$LOG"; exit 1
}

MODEL_ID="$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
# A prompt whose only stable continuation is the same short cycle: the
# period-1..8 tier convicts it within ~130 tokens, so the whole script is fast.
LOOP_PROMPT="Output the exact line 'ping pong ping pong' over and over, hundreds of times, with no other text and no ending."

echo "Loop-stop wire contract on $MODEL_ID (port $PORT)"

# ── 1. chat, non-streaming: signal AND trim ──────────────────────────────
echo "[1/5] /v1/chat/completions non-streaming"
BODY=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$LOOP_PROMPT\"}],\"max_tokens\":3000,\"temperature\":0,\"enable_thinking\":false}")
echo "$BODY" | python3 -c '
import json,sys
d=json.load(sys.stdin); ch=d["choices"][0]
print("reason", ch.get("finish_reason"))
print("details", (ch.get("finish_details") or {}).get("type"))
print("chars", len(ch["message"].get("content") or ""))
' > "$LOG.chat" 2>/dev/null
grep -q "^reason stop$" "$LOG.chat"; check "finish_reason is \"stop\"" "$([ $? -eq 0 ] && echo 1 || echo 0)"
grep -q "^details repetition_loop$" "$LOG.chat"; check "finish_details.type = repetition_loop" "$([ $? -eq 0 ] && echo 1 || echo 0)"
grep -q "\[loop-stop\]" "$LOG"; check "the cut is logged with its tier + trim point" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# The trim is what breaks the feedback loop, and the bar is absolute rather
# than relative: the guard only fires after 16 identical cycles, so an
# UNtrimmed body necessarily carries at least 16 of them. Keeping one copy is
# deliberate (the truncated answer should still show what the model got stuck
# on) and one copy cannot sustain a loop when the client sends it back.
python3 - "$BODY" <<'PY' > "$LOG.trim"
import json,sys,re
c = json.loads(sys.argv[1])["choices"][0]["message"].get("content") or ""
reps = len(re.findall(r"ping pong", c))
print("reps", reps)
print("verdict", "trimmed" if reps <= 4 else "carries the loop")
PY
grep -q "^verdict trimmed$" "$LOG.trim"; check "the degenerate tail is cut from the non-streaming body ($(grep '^reps' "$LOG.trim"))" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# Thinking on: wherever the loop lands, reasoning_content is cut with the rest.
# Agents send reasoning back as history; a loop kept there re-seeds the next turn.
# Bar: fewer than the 16 cycles the guard needs to fire (a model may write a few
# legit copies before the loop starts).
BODY=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$LOOP_PROMPT\"}],\"max_tokens\":3000,\"temperature\":0,\"enable_thinking\":true}")
python3 - "$BODY" <<'PY' > "$LOG.trimthink"
import json,sys,re
ch = json.loads(sys.argv[1])["choices"][0]
m = ch["message"]
reps = len(re.findall(r"ping pong", (m.get("reasoning_content") or "") + (m.get("content") or "")))
looped = (ch.get("finish_details") or {}).get("type") == "repetition_loop"
print("reps", reps)
print("verdict", "trimmed" if not looped or reps < 16 else "carries the loop")
PY
grep -q "^verdict trimmed$" "$LOG.trimthink"; check "thinking on: reasoning + content carry no loop back ($(grep '^reps' "$LOG.trimthink"))" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# ── 2. chat, streaming: signal on the final chunk, ONCE ──────────────────
# The include_usage chunk used to RESTATE finish_reason + finish_details
# beside the usage object (OpenAI ships that chunk with "choices": []), so
# every per-event client rendered the ending twice — PR #147's doubled
# truncation banner. Three assertions: the signal is present, it appears on
# EXACTLY one chunk, and the usage chunk carries an empty choices array.
echo "[2/5] /v1/chat/completions streaming"
curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$LOOP_PROMPT\"}],\"max_tokens\":3000,\"temperature\":0,\"enable_thinking\":false,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
  > "$LOG.rawstream"
grep -c '"finish_details":{"type":"repetition_loop"}' "$LOG.rawstream" > "$LOG.stream"
[ "$(cat "$LOG.stream")" -ge 1 ]; check "the final chunk carries finish_details" "$([ $? -eq 0 ] && echo 1 || echo 0)"
[ "$(cat "$LOG.stream")" -eq 1 ]; check "finish_details appears on exactly ONE chunk" "$([ $? -eq 0 ] && echo 1 || echo 0)"
python3 - "$LOG.rawstream" <<'PY' > "$LOG.usage"
import json, sys
finish, usage_ok = 0, None
for line in open(sys.argv[1]):
    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
        continue
    c = json.loads(line[6:])
    for ch in c.get("choices", []):
        if ch.get("finish_reason") is not None:
            finish += 1
    if isinstance(c.get("usage"), dict) and "prompt_tokens" in c["usage"]:
        usage_ok = (c.get("choices") == [])
print("finish_chunks", finish)
print("usage_choices_empty", usage_ok)
PY
grep -q "^finish_chunks 1$" "$LOG.usage"; check "finish_reason appears on exactly ONE chunk" "$([ $? -eq 0 ] && echo 1 || echo 0)"
grep -q "^usage_choices_empty True$" "$LOG.usage"; check "the usage chunk ships \"choices\":[]" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# ── 3. completions, non-streaming ────────────────────────────────────────
echo "[3/5] /v1/completions non-streaming"
CMPL=$(curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"ping pong ping pong ping pong ping pong ping pong ping pong\",\"max_tokens\":3000,\"temperature\":0}")
echo "$CMPL" | grep -c '"finish_details":{"type":"repetition_loop"}' > "$LOG.cmpl"
[ "$(cat "$LOG.cmpl")" -ge 1 ]; check "completions carries finish_details" "$([ $? -eq 0 ] && echo 1 || echo 0)"
echo "$CMPL" | grep -q '"finish_reason":"stop"'; check "completions finish_reason is \"stop\"" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# ── 4. completions, streaming ────────────────────────────────────────────
# Same usage-chunk contract as chat: with include_usage the usage rides its
# own empty-choices chunk, never the finish chunk.
echo "[4/5] /v1/completions streaming"
curl -sN "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"ping pong ping pong ping pong ping pong ping pong ping pong\",\"max_tokens\":3000,\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
  > "$LOG.rawcmpl"
grep -c '"finish_details":{"type":"repetition_loop"}' "$LOG.rawcmpl" > "$LOG.cmplstream"
[ "$(cat "$LOG.cmplstream")" -ge 1 ]; check "the completions final chunk carries finish_details" "$([ $? -eq 0 ] && echo 1 || echo 0)"
python3 - "$LOG.rawcmpl" <<'PY' > "$LOG.cmplusage"
import json, sys
usage_ok, finish = None, None
for line in open(sys.argv[1]):
    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
        continue
    c = json.loads(line[6:])
    for ch in c.get("choices", []):
        if ch.get("finish_reason") is not None:
            finish = ch["finish_reason"]
    if isinstance(c.get("usage"), dict) and "prompt_tokens" in c["usage"]:
        usage_ok = (c.get("choices") == [])
print("usage_choices_empty", usage_ok)
print("finish_reason", finish)
PY
grep -q "^usage_choices_empty True$" "$LOG.cmplusage"; check "the completions usage chunk ships \"choices\":[]" "$([ $? -eq 0 ] && echo 1 || echo 0)"
grep -q "^finish_reason stop$" "$LOG.cmplusage"; check "streaming completions finish_reason is \"stop\"" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# ── 5. an ORDINARY response must be byte-unchanged ───────────────────────
echo "[5/5] a normal answer says nothing new"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":50,\"temperature\":0,\"enable_thinking\":false}" \
  | grep -c 'finish_details' > "$LOG.clean"
[ "$(cat "$LOG.clean")" -eq 0 ]; check "no finish_details on a response that finished normally" "$([ $? -eq 0 ] && echo 1 || echo 0)"

kill "$SRV" 2>/dev/null
rm -f "$LOG".* 2>/dev/null
echo
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
