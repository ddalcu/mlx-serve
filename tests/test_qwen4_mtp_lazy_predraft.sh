#!/usr/bin/env bash
# qwen4_exp: solo greedy MTP rounds build the next draft chain from the round's lazy GPU
# results (`mtpLazyPreDraft`). At a pinned depth, greedy, against `MLX_SERVE_MTP_LAZY_PREDRAFT=0`:
#   every answer is byte-identical, lookup off and on (a lookup pick discards the lazy chain);
#   the lazy arm engages and keeps chains, and its `[mtp-trace] predraft` is far below the eager arm's;
#   a sampled request never builds one; a short answer still ends on EOS.
#   QWEN4_MODEL=<pack dir> ./tests/test_qwen4_mtp_lazy_predraft.sh [port]
set -u
MODEL="${QWEN4_MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit}"
PORT="${1:-11419}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
DIR="${OUT_DIR:-$HOME/claude-tmp/qwen4-lazy-predraft}"
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

PROMPTS="code prose repeat short needle"
body() { # $1 = prompt name, $2 = temperature
  python3 - "$1" "$MAX_TOKENS" "$2" <<'PY'
import json, sys
name, max_tokens, temp = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
filler = 'The archivist catalogued the shelves in the long hall. ' * 400
text = {
    'code': 'Write a Python function that merges two sorted lists into one sorted list, with a docstring.',
    'prose': 'Describe a quiet harbour town at dawn in two paragraphs.',
    'repeat': 'Copy this list exactly, three times: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel.',
    'short': 'What is 2 + 3? Answer with the number only.',
    'needle': filler + ' The secret code is PELICAN-42. ' + filler + ' What is the secret code? Answer with the code only.',
}[name.removeprefix('sampled-')]
print(json.dumps({'messages': [{'role': 'user', 'content': text}], 'max_tokens': max_tokens,
                  'temperature': temp, 'seed': 7, 'enable_thinking': False, 'enable_mtp': True}))
PY
}

ask() { # $1 = arm, $2 = prompt, $3 = temperature
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d "$(body "$2" "$3")" \
    | python3 -c "
import json, sys
r = json.load(sys.stdin)
open('$DIR/$1-$2.txt', 'w').write(r['choices'][0]['message']['content'])
print(r['choices'][0]['finish_reason'], r['usage']['completion_tokens'])" > "$DIR/$1-$2.meta"
}

run_arm() { # $1 = arm name, $2 = MLX_SERVE_MTP_LAZY_PREDRAFT, $3 = MLX_SERVE_MTP_LOOKUP
  local name="$1"
  local log="$DIR/$name.log"
  MLX_SERVE_MTP_LAZY_PREDRAFT="$2" MLX_SERVE_MTP_LOOKUP="$3" MLX_SERVE_MTP_FORCE_DEPTH=3 MLX_SERVE_MTP_TRACE=1 \
    "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level info --mtp --prefix-cache-entries 0 > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 600); do curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && grep -q "ready" "$log" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$log"; exit 1; }; sleep 2; done
  for p in $PROMPTS; do ask "$name" "$p" 0; done
  ask "$name" sampled-prose 0.8
  stop_srv
}

# Sum of `lazy=built/kept` over the log's spec-stats lines: prints "built kept".
lazy_counts() { grep -o 'lazy=[0-9]*/[0-9]*' "$1" | awk -F'[=/]' '{b += $2; k += $3} END {printf "%d %d", b, k}'; }
# Sum of MTP attempts over the log's spec-stats lines.
mtp_attempts() { grep -o 'mode=mtp attempts=[0-9]*' "$1" | awk -F= '{s += $NF} END {print s + 0}'; }
# Sum of lookup rounds (first field of `lookup=rounds/drafted/accepted`) over the log.
lookup_rounds() { grep -o ' lookup=[0-9]*/' "$1" | tr -dc '0-9\n' | awk '{s += $1} END {print s + 0}'; }
# Mean `predraft=` over the log's trace lines.
predraft_ms() { grep -o 'predraft=[0-9.]*' "$1" | awk -F= '{s += $2; n++} END {printf "%.3f", n ? s / n : -1}'; }

for lookup in 0 1; do
  echo "[lookup=$lookup] eager arm (MLX_SERVE_MTP_LAZY_PREDRAFT=0)"
  run_arm "eager-l$lookup" 0 "$lookup"
  echo "[lookup=$lookup] lazy arm"
  run_arm "lazy-l$lookup" 1 "$lookup"
  check "lookup=$lookup lazy engaged" "$(grep -c '\[mtp\] lazy predraft engaged' "$DIR/lazy-l$lookup.log")" "1"
  check "lookup=$lookup eager ran MTP rounds" "$([ "$(mtp_attempts "$DIR/eager-l$lookup.log")" -gt 0 ] && echo yes)" "yes"
  if [ "$lookup" = 1 ]; then
    for arm in eager lazy; do
      check "lookup=1 $arm picked lookup drafts" "$([ "$(lookup_rounds "$DIR/$arm-l1.log")" -gt 0 ] && echo yes)" "yes"
    done
  fi
  check "lookup=$lookup eager never builds" "$(lazy_counts "$DIR/eager-l$lookup.log" | cut -d' ' -f1)" "0"
  read -r built kept <<< "$(lazy_counts "$DIR/lazy-l$lookup.log")"
  check "lookup=$lookup lazy chains kept" "$([ "$kept" -gt 0 ] && echo yes)" "yes"
  echo "  info lazy built=$built kept=$kept"
  for p in $PROMPTS; do
    check "lookup=$lookup $p byte-identical to eager" "$(cmp -s "$DIR/eager-l$lookup-$p.txt" "$DIR/lazy-l$lookup-$p.txt" && echo same)" "same"
  done
  check "lookup=$lookup sampled request ran MTP" "$(grep -o 'mode=mtp attempts=[0-9]*' "$DIR/lazy-l$lookup.log" | tail -1 | awk -F= '{print ($NF > 0) ? "yes" : "no"}')" "yes"
  check "lookup=$lookup sampled request never builds" "$(grep -o 'spec-stats\] lazy=[0-9]*/[0-9]*' "$DIR/lazy-l$lookup.log" | tail -1 | grep -o 'lazy=.*')" "lazy=0/0"
  check "lookup=$lookup short answer ends on EOS" "$(cut -d' ' -f1 "$DIR/lazy-l$lookup-short.meta")" "stop"
  check "lookup=$lookup needle found" "$(grep -c 'PELICAN-42' "$DIR/lazy-l$lookup-needle.txt")" "1"
  e=$(predraft_ms "$DIR/eager-l$lookup.log")
  l=$(predraft_ms "$DIR/lazy-l$lookup.log")
  echo "  info predraft ms eager=$e lazy=$l"
  check "lookup=$lookup predraft below eager" "$(awk -v e="$e" -v l="$l" 'BEGIN {print (l >= 0 && l < e) ? "yes" : "no"}')" "yes"
done

echo "texts: $DIR"
echo "pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
