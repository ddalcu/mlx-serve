#!/bin/bash
# The degenerate-tail guard's WIRE contract, on every surface that can emit it.
#
# Live 2026-08-05 (under pi): five loop-stops in a row, each
# firing sooner than the last, surfaced to the user as "Model stopped because
# it reached the maximum output token limit" while the session's own status
# bar read 32.4%/66k. Both readings were right — `finish_reason: "length"` is
# the only value in the OpenAI schema for a server-side cut, and pi renders it
# the only way it can. The reason cannot move ("length" is what truncation
# recovery keys on — the 2026-07-14 php.html post-mortem), so the cause rides
# BESIDE it as `finish_details`.
#
# Two things are asserted, because either alone can silently regress:
#   1. the SIGNAL — `finish_details.type == "repetition_loop"` on chat
#      (stream + non-stream) and completions (stream + non-stream). A missing
#      field is invisible to any output-equality test.
#   2. the TRIM — a non-streaming response must NOT carry the degenerate tail
#      back to the client. An agent re-sends the cut turn as history, the
#      model reads its own loop and resumes it; that is the death spiral.
#      Streaming deliberately keeps the tail: a delta cannot be retracted.
#
# Needs any chat model — the prompt loops on all of them. Point LOOP_TEST_MODEL
# at a small one to keep this cheap.
#
# Usage: LOOP_TEST_MODEL=~/.mlx-serve/models/mlx-community/Qwen3.5-4B-MLX-4bit \
#          ./tests/test_loop_stop_signal.sh [port]

set -u

PORT="${1:-11266}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
MODEL="${LOOP_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-4B-MLX-4bit}"
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
for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
    kill -0 "$SRV" 2>/dev/null || break
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
    echo "[fail] server did not come up"; tail -20 "$LOG"; exit 1
}

MODEL_ID="$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
# A prompt whose only stable continuation is the same short cycle: the
# period-1..8 tier convicts it within ~130 tokens, so the whole script is fast.
LOOP_PROMPT="Output the exact line 'ping pong ping pong' over and over, hundreds of times, with no other text and no ending."

echo "Loop-stop wire contract on $MODEL_ID (port $PORT)"

# ── 1. chat, non-streaming: signal AND trim ──────────────────────────────
echo "[1/6] /v1/chat/completions non-streaming"
BODY=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$LOOP_PROMPT\"}],\"max_tokens\":3000,\"temperature\":0,\"enable_thinking\":false}")
echo "$BODY" | python3 -c '
import json,sys
d=json.load(sys.stdin); ch=d["choices"][0]
print("reason", ch.get("finish_reason"))
print("details", (ch.get("finish_details") or {}).get("type"))
print("chars", len(ch["message"].get("content") or ""))
' > "$LOG.chat" 2>/dev/null
grep -q "^reason length$" "$LOG.chat"; check "finish_reason stays \"length\"" "$([ $? -eq 0 ] && echo 1 || echo 0)"
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

# ── 2. chat, streaming: signal on the final chunk ────────────────────────
echo "[2/6] /v1/chat/completions streaming"
curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$LOOP_PROMPT\"}],\"max_tokens\":3000,\"temperature\":0,\"enable_thinking\":false,\"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
  | grep -c '"finish_details":{"type":"repetition_loop"}' > "$LOG.stream"
[ "$(cat "$LOG.stream")" -ge 1 ]; check "the final chunk carries finish_details" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# ── 3. completions, non-streaming ────────────────────────────────────────
echo "[3/6] /v1/completions non-streaming"
curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"ping pong ping pong ping pong ping pong ping pong ping pong\",\"max_tokens\":3000,\"temperature\":0}" \
  | grep -c '"finish_details":{"type":"repetition_loop"}' > "$LOG.cmpl"
[ "$(cat "$LOG.cmpl")" -ge 1 ]; check "completions carries finish_details" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# ── 4. completions, streaming ────────────────────────────────────────────
echo "[4/6] /v1/completions streaming"
curl -sN "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"ping pong ping pong ping pong ping pong ping pong ping pong\",\"max_tokens\":3000,\"temperature\":0,\"stream\":true}" \
  | grep -c '"finish_details":{"type":"repetition_loop"}' > "$LOG.cmplstream"
[ "$(cat "$LOG.cmplstream")" -ge 1 ]; check "the completions final chunk carries finish_details" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# ── 5. an ORDINARY response must be byte-unchanged ───────────────────────
echo "[5/6] a normal answer says nothing new"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":50,\"temperature\":0,\"enable_thinking\":false}" \
  | grep -c 'finish_details' > "$LOG.clean"
[ "$(cat "$LOG.clean")" -eq 0 ]; check "no finish_details on a response that finished normally" "$([ $? -eq 0 ] && echo 1 || echo 0)"

# ── 6. the trim's own kill switch, which is also red-on-revert ───────────
# Without this the trim assertion above could pass because the model happened
# to be terse. Same server, same prompt, trim disabled: the loop must come
# back. Needs its own boot — the switch is read server-side, once.
echo "[6/6] MLX_SERVE_LOOP_TRIM=0 restores the untrimmed body"
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
sleep 1
MLX_SERVE_LOOP_TRIM=0 "$BINARY" --model "$MODEL" --serve --port "$PORT" --log-file off > "$LOG.off" 2>&1 &
SRV=$!
for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
    kill -0 "$SRV" 2>/dev/null || break
done
OFF=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$LOOP_PROMPT\"}],\"max_tokens\":3000,\"temperature\":0,\"enable_thinking\":false}")
python3 - "$OFF" <<'PY' > "$LOG.off2"
import json,sys,re
c = json.loads(sys.argv[1])["choices"][0]["message"].get("content") or ""
reps = len(re.findall(r"ping pong", c))
print("reps", reps)
print("verdict", "carries the loop" if reps > 4 else "trimmed anyway")
PY
grep -q "^verdict carries the loop$" "$LOG.off2"
check "trim off = the loop is still in the body ($(grep '^reps' "$LOG.off2"))" "$([ $? -eq 0 ] && echo 1 || echo 0)"

kill "$SRV" 2>/dev/null
rm -f "$LOG".* 2>/dev/null
echo
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
