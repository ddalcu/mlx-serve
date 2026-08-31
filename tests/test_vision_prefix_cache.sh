#!/bin/bash
# Image conversations reuse the prefix cache. Vision slots were excluded from
# commit AND lookup, so every image turn re-prefilled the whole conversation.
# Two things a byte check cannot see come from the log + usage: the second
# identical request must report cached_tokens > 0 with a `[hot-cache] reused`
# line. A DIFFERENT image must never reuse state at or after its placeholder
# rows (the KV there is keyed on the pixels), but a long text prefix before
# those rows should still restore. Every answer must still name what only the
# pixels supply (the reused prefix is only correct if the restored rows are
# the image's). A Harness-style turn may append user-role context after the
# human's image message; that media still belongs to the active turn, while an
# image before the latest assistant boundary must remain historical.
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
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level debug \
  --prefix-cache-entries 4 --prefill-chunk 1024 \
  --ssm-checkpoint-stride 1024 --ssm-checkpoint-max 8 > "$LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
U="http://127.0.0.1:$PORT"
for _ in $(seq 1 600); do curl -s "$U/health" >/dev/null 2>&1 && grep -q "ready" "$LOG" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$LOG"; exit 1; }; sleep 2; done

mime_for() { case "$1" in *.jpeg|*.jpg) echo image/jpeg;; *.png) echo image/png;; *.webp) echo image/webp;; esac; }
body() { # $1 image, $2 question
  printf '{"model":"mlx-serve","max_tokens":48,"temperature":0,"enable_thinking":false,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:%s;base64,%s"}},{"type":"text","text":"%s"}]}]}' "$(mime_for "$1")" "$(base64 -i "$1")" "$2"
}
ask() { curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens_details']['cached_tokens'], '|', d['choices'][0]['message']['content'].replace(chr(10),' '))"; }
ask_prompt_tokens() { curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'])"; }
ask_anthropic() { curl -s -m 600 "$U/v1/messages" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(''.join(x.get('text','') for x in d.get('content',[]) if x.get('type') == 'text').replace(chr(10),' '))"; }
Q="What text is written on the green street signs? Answer with the words only."

injected_context_body() { # $1 image, $2 fresh|continuation
  python3 - "$1" "$2" <<'PY'
import base64, json, sys

path, mode = sys.argv[1:]
with open(path, "rb") as f:
    image_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
human = {
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "What text is written on the green street signs? Answer with the words only."},
    ],
}
context = {"role": "user", "content": "<system-reminder>Injected project context.</system-reminder>"}
messages = [human, context]
if mode == "continuation":
    messages.extend([
        {"role": "assistant", "content": "Grey Fox and Waterfall"},
        {"role": "user", "content": "Reply with CONTINUATION only."},
        context,
    ])
elif mode == "assistant-prefix":
    messages = [human, {"role": "assistant", "content": "The green street signs read "}]
payload = {
    "model": "mlx-serve", "max_tokens": 48, "temperature": 0,
    "enable_thinking": False, "messages": messages,
}
if mode == "assistant-prefix":
    payload["continue_final_message"] = True
print(json.dumps(payload))
PY
}

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

echo "[5] image before trailing user-role context is processed"
mm_before=$(grep -c 'Multimodal: processing' "$LOG" || true)
mrope_before=$(grep -c 'M-RoPE: 1 images' "$LOG" || true)
r0=$(injected_context_body "$F1" fresh | ask); echo "  $r0"
mm_after=$(grep -c 'Multimodal: processing' "$LOG" || true)
mrope_after=$(grep -c 'M-RoPE: 1 images' "$LOG" || true)
check "injected-context turn processes one image" "$((mm_after-mm_before))" "1"
check "injected-context answer names the sign text" "$(echo "$r0" | grep -ciE 'gr[ae]y fox|waterfall' | sed 's/^[1-9][0-9]*$/1/')" "1"
check "injected-context turn builds Qwen M-RoPE" "$((mrope_after-mrope_before))" "1"

echo "[6] text continuation does not reprocess historical media"
mm_before=$mm_after
decode_before=$(grep -c 'Decoded .* image' "$LOG" || true)
r0c=$(injected_context_body "$F1" continuation | ask); echo "  $r0c"
mm_after=$(grep -c 'Multimodal: processing' "$LOG" || true)
decode_after=$(grep -c 'Decoded .* image' "$LOG" || true)
check "historical image stays behind assistant boundary" "$((mm_after-mm_before))" "0"
check "historical image is not decoded during parsing" "$((decode_after-decode_before))" "0"
check "text-only continuation completes" "$(echo "$r0c" | grep -cE '\| .+' | sed 's/^[1-9][0-9]*$/1/')" "1"

historical_many_body() { # $1 image, $2 count
  python3 - "$1" "$2" <<'PY'
import base64, json, sys

path, count = sys.argv[1], int(sys.argv[2])
with open(path, "rb") as f:
    image_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
messages = []
for i in range(count):
    messages.extend([
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": f"Historical image {i + 1}."},
        ]},
        {"role": "assistant", "content": "Seen."},
    ])
messages.append({"role": "user", "content": "Reply with CURRENT only."})
print(json.dumps({
    "model": "mlx-serve", "max_tokens": 8, "temperature": 0,
    "enable_thinking": False, "messages": messages,
}))
PY
}

anthropic_historical_body() { # $1 image
  python3 - "$1" <<'PY'
import base64, json, sys

with open(sys.argv[1], "rb") as f:
    image_data = base64.b64encode(f.read()).decode()
print(json.dumps({
    "model": "mlx-serve", "max_tokens": 8, "temperature": 0,
    "messages": [
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
            {"type": "text", "text": "Historical image."},
        ]},
        {"role": "assistant", "content": "Seen."},
        {"role": "user", "content": "Reply with CURRENT only."},
    ],
}))
PY
}

image_only_history_body() { # $1 image, $2 image|empty
  python3 - "$1" "$2" <<'PY'
import base64, json, sys

path, mode = sys.argv[1:]
with open(path, "rb") as f:
    image_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
first_content = ([{"type": "image_url", "image_url": {"url": image_url}}]
                 if mode == "image" else "")
print(json.dumps({
    "model": "mlx-serve", "max_tokens": 4, "temperature": 0,
    "enable_thinking": False,
    "messages": [
        {"role": "user", "content": first_content},
        {"role": "assistant", "content": "Seen."},
        {"role": "user", "content": "Reply with CURRENT only."},
    ],
}))
PY
}

echo "[7] twenty historical images remain lazy on a text-only turn"
decode_before=$decode_after
r0h=$(historical_many_body "$F1" 20 | ask); echo "  $r0h"
decode_after=$(grep -c 'Decoded .* image' "$LOG" || true)
check "twenty historical images trigger zero decodes" "$((decode_after-decode_before))" "0"
check "large historical-media request completes" "$(echo "$r0h" | grep -cE '\| .+' | sed 's/^[1-9][0-9]*$/1/')" "1"

echo "[8] an image-only historical user turn remains in the rendered prompt"
decode_before=$decode_after
image_only_tokens=$(image_only_history_body "$F1" image | ask_prompt_tokens)
dropped_empty_tokens=$(image_only_history_body "$F1" empty | ask_prompt_tokens)
decode_after=$(grep -c 'Decoded .* image' "$LOG" || true)
check "image-only history triggers zero decodes" "$((decode_after-decode_before))" "0"
check "image-only user boundary contributes prompt tokens" "$(python3 -c "print(1 if int('$image_only_tokens') > int('$dropped_empty_tokens') else 0)")" "1"

echo "[9] Anthropic text continuation also leaves historical images lazy"
decode_before=$decode_after
r0a=$(anthropic_historical_body "$F1" | ask_anthropic); echo "  $r0a"
decode_after=$(grep -c 'Decoded .* image' "$LOG" || true)
check "Anthropic historical image triggers zero decodes" "$((decode_after-decode_before))" "0"
check "Anthropic text-only continuation completes" "$(echo "$r0a" | grep -cE '.+' | sed 's/^[1-9][0-9]*$/1/')" "1"

echo "[10] assistant-prefix continuation keeps current-turn media"
mm_before=$mm_after
r0p=$(injected_context_body "$F1" assistant-prefix | ask); echo "  $r0p"
mm_after=$(grep -c 'Multimodal: processing' "$LOG" || true)
check "assistant-prefix continuation processes one image" "$((mm_after-mm_before))" "1"
check "assistant-prefix continuation reads the signs" "$(echo "$r0p" | grep -ciE 'gr[ae]y fox|waterfall' | sed 's/^[1-9][0-9]*$/1/')" "1"

openai_parser_gap_body() { # $1 image, $2 empty-assistant|trailing-empty-user
  python3 - "$1" "$2" <<'PY'
import base64, json, sys

path, mode = sys.argv[1:]
with open(path, "rb") as f:
    image_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
human = {"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": image_url}},
    {"type": "text", "text": "What text is written on the green street signs? Answer with the words only."},
]}
if mode == "empty-assistant":
    messages = [human, {"role": "assistant", "content": ""}, {"role": "user", "content": "Answer the image question."}]
else:
    messages = [human, {"role": "assistant", "content": "The green street signs read "}, {"role": "user", "content": ""}]
payload = {
    "model": "mlx-serve", "max_tokens": 48, "temperature": 0,
    "enable_thinking": False, "messages": messages,
}
if mode == "trailing-empty-user":
    payload["continue_final_message"] = True
print(json.dumps(payload))
PY
}

echo "[11] an empty OpenAI assistant skipped by parsing is not a media boundary"
decode_before=$(grep -c 'Decoded .* image' "$LOG" || true)
r0e=$(openai_parser_gap_body "$F1" empty-assistant | ask); echo "  $r0e"
decode_after=$(grep -c 'Decoded .* image' "$LOG" || true)
check "empty assistant still decodes the active image" "$((decode_after-decode_before))" "1"
check "empty assistant response reads the signs" "$(echo "$r0e" | grep -ciE 'gr[ae]y fox|waterfall' | sed 's/^[1-9][0-9]*$/1/')" "1"

echo "[12] a trailing empty OpenAI user skipped by parsing preserves continuation"
decode_before=$decode_after
r0u=$(openai_parser_gap_body "$F1" trailing-empty-user | ask); echo "  $r0u"
decode_after=$(grep -c 'Decoded .* image' "$LOG" || true)
check "trailing empty user still decodes the active image" "$((decode_after-decode_before))" "1"
check "trailing empty user continues with the sign text" "$(echo "$r0u" | grep -ciE 'gr[ae]y fox|waterfall' | sed 's/^[1-9][0-9]*$/1/')" "1"

# Growing image conversations move the current image span on every turn. A
# hybrid cache entry may have the longest raw token match at the previous image
# boundary while its first SSM checkpoint sits just beyond that boundary. An
# older entry with a slightly shorter match can still restore safely. The old
# longest-raw-match policy produced cached-token counts 0,2048,0,2048 here;
# selecting by the restorable checkpoint keeps every continuation warm.
conversation_body() { # $1 image, $2 turn count
  python3 - "$1" "$2" <<'PY'
import base64, json, sys

path, turns = sys.argv[1], int(sys.argv[2])
with open(path, "rb") as f:
    image_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
messages = [{"role": "system", "content": "alpha " * 2600}]
for turn in range(1, turns + 1):
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": f"Turn {turn}: reply with OK only."},
        ],
    })
    if turn < turns:
        messages.append({"role": "assistant", "content": "OK"})
print(json.dumps({
    "model": "mlx-serve", "max_tokens": 4, "temperature": 0,
    "enable_thinking": False, "enable_mtp": False, "messages": messages,
}))
PY
}

echo "[13] same image across a growing conversation: every continuation restores"
for turn in 1 2 3 4; do
  r=$(conversation_body "$F1" "$turn" | ask); echo "  turn $turn: $r"
  if [ "$turn" -gt 1 ]; then
    check "growing image turn $turn: cached_tokens > 0" "$(python3 -c "print(1 if int('${r%% |*}')>0 else 0)")" "1"
  fi
done

long_prefix_body() { # $1 image
  python3 - "$1" <<'PY'
import base64, json, sys

with open(sys.argv[1], "rb") as f:
    image_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
print(json.dumps({
    "model": "mlx-serve", "max_tokens": 4, "temperature": 0,
    "enable_thinking": False,
    "messages": [
        {"role": "system", "content": "beta " * 2600},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "Reply with OK only."},
        ]},
    ],
}))
PY
}

# F1 commits a >2K text prefix followed by its image rows. Switching to F2
# must recover the 2048 checkpoint before those rows, while the short F2 entry
# from [3] has no useful checkpoint. Exact vision-key filtering made this cold.
echo "[14] changed image reuses only the long text prefix before media"
r9a=$(long_prefix_body "$F1" | ask); echo "  first image: $r9a"
r9b=$(long_prefix_body "$F2" | ask); echo "  changed image: $r9b"
check "changed image: cached_tokens > 0 before media" "$(python3 -c "print(1 if int('${r9b%% |*}')>0 else 0)")" "1"

echo "pass=$pass fail=$fail"
[ "$fail" = "0" ] && echo "PASS: vision prefix cache" || { echo "FAIL: vision prefix cache"; grep -E "hot-cache|cache\]" "$LOG" | tail -20; }
[ "$fail" = "0" ]
