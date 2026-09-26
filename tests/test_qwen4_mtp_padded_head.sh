#!/usr/bin/env bash
# qwen4_exp: the MTP head appends the whole verify row as history and drafts past its dead
# rows (`Transformer.HeadPlace`). Asserts, at a pinned depth, greedy:
#   the padded path engages (and `MLX_SERVE_MTP_PADDED_HEAD=0` does not);
#   every answer reruns byte-identical and reads as text;
#   a short answer still ends on EOS, and a long needle prompt (head QSA engaged) finds its needle.
# The texts land in $DIR/<arm>-<prompt>-<rep>.txt, the padded-off arm beside them for reading.
#   QWEN4_MODEL=<pack dir> ./tests/test_qwen4_mtp_padded_head.sh [port]
set -u
MODEL="${QWEN4_MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit}"
PORT="${1:-11417}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
DIR="${OUT_DIR:-$HOME/claude-tmp/qwen4-padded-head}"
MAX_TOKENS=200
mkdir -p "$DIR"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no pack at $MODEL"; exit 0; }
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }

SPID=""
stop_srv() {
  [ -n "${SPID:-}" ] || return 0
  kill "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null
  SPID=""
}
trap stop_srv EXIT INT TERM

PROMPTS="code prose short needle"
body() { # $1 = prompt name
  python3 - "$1" "$MAX_TOKENS" <<'PY'
import json, sys
name, max_tokens = sys.argv[1], int(sys.argv[2])
filler = 'The archivist catalogued the shelves in the long hall. ' * 400
text = {
    'code': 'Write a Python function that merges two sorted lists into one sorted list, with a docstring.',
    'prose': 'Describe a quiet harbour town at dawn in two paragraphs.',
    'short': 'What is 2 + 3? Answer with the number only.',
    'needle': filler + ' The secret code is PELICAN-42. ' + filler + ' What is the secret code? Answer with the code only.',
}[name]
print(json.dumps({'messages': [{'role': 'user', 'content': text}], 'max_tokens': max_tokens,
                  'temperature': 0, 'enable_thinking': False, 'enable_mtp': True}))
PY
}

ask() { # $1 = arm, $2 = prompt, $3 = rep; writes the text and "finish_reason completion_tokens"
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d "$(body "$2")" \
    | python3 -c "
import json, sys
r = json.load(sys.stdin)
open('$DIR/$1-$2-$3.txt', 'w').write(r['choices'][0]['message']['content'])
print(r['choices'][0]['finish_reason'], r['usage']['completion_tokens'])" > "$DIR/$1-$2-$3.meta"
}

run_arm() { # $1 = arm name, $2 = MLX_SERVE_MTP_PADDED_HEAD, $3 = reps
  local name="$1"
  local log="$DIR/$name.log"
  MLX_SERVE_MTP_PADDED_HEAD="$2" MLX_SERVE_MTP_FORCE_DEPTH=3 "$BIN" --model "$MODEL" --serve --host 127.0.0.1 \
    --port "$PORT" --log-level info --mtp --prefix-cache-entries 0 > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 600); do curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && grep -q "ready" "$log" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$log"; exit 1; }; sleep 2; done
  for p in $PROMPTS; do
    for rep in $(seq 1 "$3"); do ask "$name" "$p" "$rep"; done
  done
  stop_srv
}

echo "[1] padded head (default)"
run_arm padded 1 2
# Sum of MTP attempts over the log's spec-stats lines (a lookup-only line logs attempts=0).
mtp_attempts() { grep -o 'mode=mtp attempts=[0-9]*' "$1" | awk -F= '{s += $NF} END {print s + 0}'; }
check "padded path engaged" "$(grep -c '\[mtp\] padded head history engaged' "$DIR/padded.log")" "1"
check "MTP rounds ran" "$([ "$(mtp_attempts "$DIR/padded.log")" -gt 0 ] && echo yes)" "yes"
for p in $PROMPTS; do
  check "$p reruns byte-identical" "$(cmp -s "$DIR/padded-$p-1.txt" "$DIR/padded-$p-2.txt" && echo same)" "same"
  check "$p answered" "$([ -s "$DIR/padded-$p-1.txt" ] && echo yes)" "yes"
  check "$p has no replacement characters" "$(grep -c $'\xef\xbf\xbd' "$DIR/padded-$p-1.txt")" "0"
  check "$p stays within max_tokens" "$(awk -v m="$MAX_TOKENS" '{print ($2 <= m) ? "yes" : "no"}' "$DIR/padded-$p-1.meta")" "yes"
done
check "short answer ends on EOS" "$(cut -d' ' -f1 "$DIR/padded-short-1.meta")" "stop"
check "short answer is 5" "$(grep -c '5' "$DIR/padded-short-1.txt" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "needle found past the head's QSA budget" "$(grep -c 'PELICAN-42' "$DIR/padded-needle-1.txt")" "1"

echo "[2] MLX_SERVE_MTP_PADDED_HEAD=0 keeps the merged step"
run_arm merged 0 1
check "padded path off" "$(grep -c '\[mtp\] padded head history engaged' "$DIR/merged.log")" "0"
check "merged arm ran MTP rounds" "$([ "$(mtp_attempts "$DIR/merged.log")" -gt 0 ] && echo yes)" "yes"
for p in $PROMPTS; do
  if cmp -s "$DIR/padded-$p-1.txt" "$DIR/merged-$p-1.txt"; then echo "  info $p: padded == merged"; else echo "  info $p: padded and merged differ (drafts move acceptance, like a width change)"; fi
done
grep -h 'spec-stats' "$DIR/padded.log" "$DIR/merged.log" | tail -8

echo "texts: $DIR"
echo "pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
