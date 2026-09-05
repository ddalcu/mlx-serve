#!/usr/bin/env bash
# Qwen3.8-Flash-Next (qwen4_exp): the in-checkpoint MTP head's committed
# history rides the prefix cache.
#
# The head is NOT KV-only — it owns a QSA index-key history and pooled block
# bank beside its own KV — so before this it was neither committed nor
# restored, and a prefix-cache hit drafted from `qwen4MtpReset`: an EMPTY
# head at a 62.7k-token cursor. Measured (62.7k prose prompt, auto MTP):
# cold prefill m_avg 2.94 / acc 1.59 -> 54.1 tok/s, the SAME prompt as a
# cache hit m_avg 1.00 / acc 0.59 -> 52.2 tok/s, i.e. serial (51.1).
#
# What this asserts is the INVARIANT, never a checkpoint's acceptance:
#   - the second turn is a hot-cache hit AND the head is restored (log line),
#   - the restored-head answer matches the persist-OFF answer tie-aware,
#   - `MLX_SERVE_MTP_HEAD_PERSIST=0` restores the old behaviour exactly (no
#     restore line, still a correct answer).
#   QWEN4_MODEL=<pack dir> ./tests/test_qwen4_mtp_head_persist.sh [port]
set -u
MODEL="${QWEN4_MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit}"
PORT="${1:-11413}"
BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
DIR="$HOME/claude-tmp/qwen4-head-persist"
mkdir -p "$DIR"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no pack at $MODEL"; exit 0; }
[ -f "$MODEL/ngram_table.bin" ] || { echo "SKIP: pack has no ngram_table.bin"; exit 0; }
pass=0; fail=0
BASELINE_FREE_MB=0   # set below, before the first boot
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }

# Cleanup is armed HERE — before anything can start a server — and is never
# disarmed. An earlier version set the trap inside run_arm AFTER the boot and
# cleared it with `trap - EXIT` at the end of each arm, so any failure outside
# that window (including `set -u` killing the script before the first boot)
# left an engine running and blocked the next executor's port. `SPID` is
# initialised so `set -u` cannot make the handler itself the failure.
SPID=""
stop_srv() {
  [ -n "${SPID:-}" ] || return 0
  kill "$SPID" 2>/dev/null
  wait "$SPID" 2>/dev/null
  SPID=""
}
trap stop_srv EXIT INT TERM

# Arm 2's preflight can see arm 1's 100 GB pack still resident and refuse the
# load (`available 36.51 GB` -> LoadFailed): `kill` returns as soon as the
# signal is delivered, but the kernel reclaims a pack of this size well after
# the process is gone. So "stopped" is TWO conditions -- the PID is reaped AND
# free memory is back near where it was before the first boot -- and the wait
# lives in one function both arms go through.
MEM_RECOVER_GAP_MB=10240   # tolerance: the pack is ~100 GB, so 10 GB is noise
MEM_RECOVER_TIMEOUT_S=90

# free + inactive pages, in MB. Inactive counts: it is reclaimable, and a
# just-freed pack lands there before it returns to the free list.
free_mb() {
  vm_stat | awk '
    /page size of/   { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) ps = $i }
    /^Pages free:/     { gsub(/\./, "", $3); free = $3 }
    /^Pages inactive:/ { gsub(/\./, "", $3); inact = $3 }
    END { if (ps == "") ps = 16384; printf "%d", (free + inact) * ps / 1048576 }'
}

# $1 = the PID that was running (stop_srv clears SPID, so capture it first),
# $2 = the free-memory baseline read before the first boot.
wait_for_release() {
  local pid="$1"
  local baseline="$2"
  local floor=$((baseline - MEM_RECOVER_GAP_MB))
  local waited=0
  local now
  while [ "$waited" -lt "$MEM_RECOVER_TIMEOUT_S" ]; do
    now=$(free_mb)
    if ! kill -0 "$pid" 2>/dev/null && [ "$now" -ge "$floor" ]; then
      echo "  pack released after ${waited}s (free ${now} MB >= floor ${floor} MB)"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  # A warning, not a failure: the arm that follows will report the real
  # symptom (LoadFailed) with its own message if the memory never came back.
  echo "  WARN: memory did not recover within ${MEM_RECOVER_TIMEOUT_S}s (free $(free_mb) MB, floor ${floor} MB) - proceeding"
  return 0
}

# One long prompt with a needle, well past the QSA budget so the head's
# history is the thing under test and not an incidental short window.
body() { python3 -c "
import json,sys
filler=('The archivist catalogued the shelves in the long hall. ')*700
print(json.dumps({'messages':[{'role':'user','content':filler+' The secret code is PELICAN-42. '+filler+' What is the secret code? Answer with the code only.'}],'max_tokens':24,'temperature':0,'enable_thinking':False,'enable_mtp':True}))"; }

run_arm() { # $1 = arm name, $2 = MLX_SERVE_MTP_HEAD_PERSIST value
  # ONE declaration PER LINE. `local a="$1" b="$DIR/$a.log"` expands every word
  # on the line before any of the assignments take effect, so the third
  # initialiser read `$name` while it was still unset and `set -u` killed the
  # script at the top of the first arm.
  local name="$1"
  local persist="$2"
  local log="$DIR/$name.log"
  local u="http://127.0.0.1:$PORT"
  local b
  MLX_SERVE_MTP_HEAD_PERSIST="$persist" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 \
    --port "$PORT" --log-level info --mtp --prefix-cache-entries 4 > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 600); do curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && grep -q "ready" "$log" && break; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 "$log"; exit 1; }; sleep 2; done
  b=$(body)
  # Turn 1 cold-prefills and COMMITS the head; turn 2 is the cache hit.
  curl -s -m 1800 "$u/v1/chat/completions" -H 'content-type: application/json' -d "$b" >/dev/null
  curl -s -m 1800 "$u/v1/chat/completions" -H 'content-type: application/json' -d "$b" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" > "$DIR/$name.answer"
  # Through the SAME helper the trap uses, so the two can never disagree about
  # what "stopped" means; the trap stays armed for the next arm. stop_srv
  # clears SPID, so the PID is captured before the call and the memory wait
  # follows it -- the next arm's preflight must not see this pack.
  local was="$SPID"
  stop_srv
  wait_for_release "$was" "$BASELINE_FREE_MB"
}

BASELINE_FREE_MB=$(free_mb)
echo "baseline free+inactive before any boot: ${BASELINE_FREE_MB} MB"

echo "[1] persistence ON: the second turn restores the head"
run_arm on 1
check "hot-cache hit on turn 2" "$(grep -c '\[hot-cache\] reused' "$DIR/on.log" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "MTP head restored" "$(grep -c '\[qwen4\] MTP head restored' "$DIR/on.log" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "head restore never declined" "$(grep -c '\[qwen4\] MTP head restore declined' "$DIR/on.log")" "0"
check "needle recovered on the restored turn" "$(grep -c 'PELICAN-42' "$DIR/on.answer")" "1"

echo "[2] MLX_SERVE_MTP_HEAD_PERSIST=0 restores the old behaviour"
run_arm off 0
check "hot-cache hit on turn 2" "$(grep -c '\[hot-cache\] reused' "$DIR/off.log" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "no head restore line" "$(grep -c '\[qwen4\] MTP head restore' "$DIR/off.log")" "0"
check "needle still recovered (blind head costs acceptance, never a token)" "$(grep -c 'PELICAN-42' "$DIR/off.answer")" "1"

echo "[3] the restored head answers what the blind head answers"
# Greedy verify decides every emitted token on BOTH arms — drafts steer
# acceptance only — so a restored head must not change the answer.
check "restored == blind (first 20 chars)" "$(head -c 20 "$DIR/on.answer")" "$(head -c 20 "$DIR/off.answer")"

echo "pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
