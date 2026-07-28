#!/usr/bin/env bash
# Long-context decode must not ratchet the process footprint (issue #110).
#
# The class: MLX parks freed buffers in a size-keyed pool instead of returning
# them to the OS, and its own trim limits are ~121 GB / ~91 GB on a 128 GB Mac
# (`backend/metal/allocator.cpp`) — so anything that frees never-repeating sizes
# in a loop grows the footprint unbounded. Two defects fed that pool:
#
#   * the KV cache grew by a fixed 256 tokens, orphaning a whole capacity-sized
#     buffer every 256 generated tokens (~5.6 GB each at the reporter's 89 K
#     context on a 64 KiB/token model);
#   * the drafter / MTP / batched decode paths never called `mlx_clear_cache()`
#     at ALL — and a `-mtp` checkpoint on a dense trunk defaults onto one.
#
# The reporter's process reached 81.4 GB while the app panel read 19.6 GB. That
# gap is the whole bug, so the assertions are written against the gap:
# `cache_bytes` (now served by /props) and `memory_mb - active_bytes`.
#
# `[spec-stats] mode=mtp` is asserted in the server log — without it this script
# silently measures the wrong decode path, which is the same class as the two
# hardcoded `use_drafter=false` call sites that shipped for a month.
#
# Skips gracefully when the checkpoint isn't downloaded.
# Usage: KV_MEM_MODEL=<dir> ./tests/test_kv_cache_growth_memory.sh [port]
#        KV_MEM_FULL=1 ...   reproduce the reporter's shape verbatim (~10 min)
set -uo pipefail
PORT="${1:-11419}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
MODEL="${KV_MEM_MODEL:-$HOME/.mlx-serve/models/ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve}"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no checkpoint at $MODEL"; exit 0; }

# Default shape keeps a full run near 3 minutes while still crossing several KV
# growth events per turn. KV_MEM_FULL=1 is the reporter's own: 120 K ctx, ~85 K
# prompt.
# NOTE the word count is not the token count: `w123` costs ~4 tokens, so 6000
# words ≈ 23 K tokens. Sized so turn 3's grown prompt plus MAX_TOKENS still
# clears CTX — an over-ctx prompt is an honest 400 and measures nothing.
if [ "${KV_MEM_FULL:-0}" = "1" ]; then
  CTX=120000; PROMPT_WORDS=20000; GROW_WORDS=2500; MAX_TOKENS=1500
else
  CTX=40000;  PROMPT_WORDS=6000;  GROW_WORDS=750;  MAX_TOKENS=1200
fi

LOG=/tmp/test_kv_cache_growth_memory_server.log
: >"$LOG"
# --prefix-cache-entries 0: a warm hot-cache would keep the KV buffers of a
# previous turn resident and confuse "what did THIS turn strand".
"$BIN" --serve --port "$PORT" --model "$MODEL" --ctx-size "$CTX" \
       --metrics --prefix-cache-entries 0 >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -20 "$LOG"; exit 1; }
  sleep 1
done

props() { # field -> bytes, -1 on failure
  curl -s "http://127.0.0.1:$PORT/props" | python3 -c "
import json,sys
try: print(json.load(sys.stdin)['memory'].get('$1', -1))
except Exception: print(-1)"
}
footprint() { # process phys_footprint in bytes, via the metrics gauge
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | awk '/^mlx_serve:memory_mb /{printf "%d\n", $2 * 1048576; found=1} END{if(!found) print -1}'
}
gb() { python3 -c "print(f'{$1/1073741824:.2f}')"; }

# One long prompt built client-side. Each turn appends to it, so the context
# GROWS across turns exactly like an agent session — which is what makes the KV
# buffer cross growth events instead of sitting in one allocation.
mkprompt() { # words
  python3 -c "
import sys
n = int(sys.argv[1])
print(' '.join(f'w{i%997}' for i in range(n)))" "$1"
}

chat() { # prompt-file -> http code
  python3 - "$1" "$PORT" "$MAX_TOKENS" <<'PY'
import json, sys, urllib.request
path, port, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = json.dumps({
    "model": "local",
    "messages": [
        {"role": "user", "content": open(path).read()},
        {"role": "user", "content": "Count slowly from one to four hundred in words, one number per line."},
    ],
    "max_tokens": max_tokens,
    "temperature": 0.0,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        json.load(r)
        print(200)
except Exception as e:
    print(getattr(e, "code", 0))
PY
}

# ── budget ──
# cache_bytes: the cap is 8 GB on a 128 GB Mac; 10 GB leaves room for the moment
# between a large free and the next allocation. Pre-fix this reads tens of GB.
CACHE_MAX=$((10 * 1024 * 1024 * 1024))
# memory_mb - active_bytes: the gap the reporter screenshotted was ~61 GB.
GAP_MAX=$((12 * 1024 * 1024 * 1024))
# footprint ratchet across the three turns.
RATCHET_MAX=$((2 * 1024 * 1024 * 1024))

sample() { # label
  local a c f gap
  a=$(props active_bytes); c=$(props cache_bytes); f=$(footprint)
  [ "$c" != "-1" ] || { echo "FAIL: /props does not report memory.cache_bytes"; exit 1; }
  [ "$f" != "-1" ] || { echo "FAIL: /metrics does not report mlx_serve:memory_mb"; exit 1; }
  gap=$((f - a)); [ "$gap" -ge 0 ] || gap=0
  printf '%-22s active %6s GB   cache %6s GB   footprint %6s GB   gap %6s GB\n' \
    "$1" "$(gb "$a")" "$(gb "$c")" "$(gb "$f")" "$(gb "$gap")"
  [ "$c" -le "$CACHE_MAX" ] || { echo "FAIL: $1: MLX buffer pool $(gb "$c") GB exceeds $(gb "$CACHE_MAX") GB"; exit 1; }
  [ "$gap" -le "$GAP_MAX" ] || { echo "FAIL: $1: footprint is $(gb "$gap") GB above active_bytes (the #110 gap)"; exit 1; }
  LAST_FOOTPRINT=$f
}

PROMPT=/tmp/test_kv_cache_growth_memory_prompt.txt
mkprompt "$PROMPT_WORDS" >"$PROMPT"

sample "idle (loaded)"

BASE_FOOTPRINT=""
for turn in 1 2 3; do
  code=$(chat "$PROMPT")
  [ "$code" = "200" ] || { echo "FAIL: turn $turn -> HTTP $code"; tail -20 "$LOG"; exit 1; }
  sample "after turn $turn"
  # Turn 1 pays the one-time costs a ratchet check must not bill: the first
  # prefill's KV, and the buffer pool warming to its working size. The ratchet
  # is what turns 2 and 3 ADD on top of that steady state.
  [ -n "$BASE_FOOTPRINT" ] || BASE_FOOTPRINT=$LAST_FOOTPRINT
  # Grow the prompt so the next turn's KV crosses more growth events.
  mkprompt "$GROW_WORDS" >>"$PROMPT"
done

RATCHET=$((LAST_FOOTPRINT - BASE_FOOTPRINT))
echo "footprint ratchet, turn 1 -> turn 3: $(gb "$RATCHET") GB"
[ "$RATCHET" -le "$RATCHET_MAX" ] \
  || { echo "FAIL: footprint ratcheted $(gb "$RATCHET") GB across three turns"; exit 1; }

# ── engagement ──
# Without this the script is measuring `Generator.next`, which ALWAYS cleared —
# i.e. it would pass green against the exact defect it exists to catch.
ENGAGED=$(grep -c '\[spec-stats\] mode=mtp' "$LOG")
[ "$ENGAGED" -gt 0 ] \
  || { echo "FAIL: no '[spec-stats] mode=mtp' in the server log — the MTP decode path never ran, so this run measured nothing"; exit 1; }
echo "PASS: MTP decode path engaged ($ENGAGED rounds logged)"

echo "ALL PASS: long-context decode does not ratchet the MLX buffer pool ($(basename "$MODEL"))"
