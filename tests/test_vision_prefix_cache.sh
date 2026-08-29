#!/bin/bash
# Image conversations reuse the prefix cache. Vision slots were excluded from
# commit AND lookup, so every image turn re-prefilled the whole conversation.
# Two things a byte check cannot see come from the log + usage: the second
# identical request must report cached_tokens > 0 with a `[hot-cache] reused`
# line, and a DIFFERENT image under the SAME placeholder tokens must NOT hit
# (the KV is keyed on the pixels, `vision_key`). Every answer must still name
# what only the pixels supply (the reused prefix is only correct if the
# restored rows are the image's).
set -u
MODEL="${VISION_CACHE_MODEL:-${1:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}}"
PORT="${2:-11419}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
LOG="$HOME/claude-tmp/vision-cache/server-$PORT.log"
mkdir -p "$(dirname "$LOG")"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no model at $MODEL"; exit 0; }
F1="tests/fixtures/street-name-signs.jpg"
F2="tests/fixtures/house.jpeg"
for f in "$F1" "$F2"; do [ -f "$f" ] || { echo "SKIP: fixture $f missing"; exit 0; }; done
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level debug --prefix-cache-entries 4 > "$LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
U="http://127.0.0.1:$PORT"
for _ in $(seq 1 600); do curl -s "$U/health" >/dev/null 2>&1 && grep -q "ready" "$LOG" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$LOG"; exit 1; }; sleep 2; done

mime_for() { case "$1" in *.jpeg|*.jpg) echo image/jpeg;; *.png) echo image/png;; *.webp) echo image/webp;; esac; }
body() { # $1 image, $2 question
  printf '{"model":"mlx-serve","max_tokens":48,"temperature":0,"enable_thinking":false,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:%s;base64,%s"}},{"type":"text","text":"%s"}]}]}' "$(mime_for "$1")" "$(base64 -i "$1")" "$2"
}
ask() { curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens_details']['cached_tokens'], '|', d['choices'][0]['message']['content'].replace(chr(10),' '))"; }
Q="What text is written on the green street signs? Answer with the words only."

echo "[1] cold image turn"
r1=$(body "$F1" "$Q" | ask); echo "  $r1"
check "answer names the sign text" "$(echo "$r1" | grep -ciE 'gr[ae]y fox|waterfall' | sed 's/^[1-9][0-9]*$/1/')" "1"
check "cold: cached_tokens 0" "${r1%% |*}" "0"

echo "[2] identical image turn: prefix hit"
r2=$(body "$F1" "$Q" | ask); echo "  $r2"
check "warm: cached_tokens > 0" "$(python3 -c "print(1 if int('${r2%% |*}')>0 else 0)")" "1"
check "hot-cache reused line" "$(grep -c 'hot-cache\] reused' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "warm answer still names the sign text" "$(echo "$r2" | grep -ciE 'gr[ae]y fox|waterfall' | sed 's/^[1-9][0-9]*$/1/')" "1"
# No byte-equality: a prefix-cache HIT is not bit-identical on a hybrid (the
# restore lands at a checkpoint and re-prefills the tail; measured one token).

echo "[3] different image, same question: no hit on foreign pixels"
r3=$(body "$F2" "What is the main subject of this picture? One short sentence." | ask); echo "  $r3"
check "foreign image: cached_tokens 0" "${r3%% |*}" "0"
check "answer describes the house, not the signs" "$(echo "$r3" | grep -ciE 'house|home|building' | sed 's/^[1-9][0-9]*$/1/')" "1"

echo "[4] same image, a different question: the image span restores, the tail prefills"
r4=$(body "$F1" "What shape is the red sign? One word." | ask); echo "  $r4"
check "same-image follow-up: cached_tokens > 0" "$(python3 -c "print(1 if int('${r4%% |*}')>0 else 0)")" "1"
check "follow-up answer reads the stop sign" "$(echo "$r4" | grep -ciE 'octagon|stop' | sed 's/^[1-9][0-9]*$/1/')" "1"

echo "pass=$pass fail=$fail"
[ "$fail" = "0" ] && echo "PASS: vision prefix cache" || { echo "FAIL: vision prefix cache"; grep -E "hot-cache|cache\]" "$LOG" | tail -20; }
[ "$fail" = "0" ]
