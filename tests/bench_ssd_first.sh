#!/bin/bash
# SSD-first prefix cache — measurement harness (qwen4_exp).
#
# EDIT-ONLY UNTIL A GPU SLOT IS GRANTED. This boots a server; do not run it
# while another arm owns the engine.
#
# Answers the five questions the design set, per rung, for BOTH arms
# (SSD-first on, and `MLX_SERVE_PREFIX_SSD_FIRST=0` as the control):
#
#   1. TTFT for a COLD prefill, a RAM hit (same session), and a DISK hit
#      (the session that was pushed out by the other chain).
#   2. Idle residency: `/props` active bytes with no request in flight, which
#      the design says must settle at weights + ONE session.
#   3. Disk bytes written per turn and the flush's own wall time, off the
#      `[disk-cache] persisted` line.
#   4. The stall a pending flush imposes on the NEXT request: a short probe is
#      fired immediately behind each long turn and its TTFT is recorded, so a
#      writer that is secretly synchronous shows up as a spike.
#   5. kill -9 durability: SIGKILL during a flush, restart, and report what
#      survived (a chunk-aligned prefix must restore; a half-indexed entry must
#      not exist).
#
# Two interleaved prompt chains are what make (1) meaningful: with RAM holding
# one session, chain B's turn evicts chain A to disk, so A's next turn is a
# genuine disk hit rather than a RAM hit wearing a different name.
#
# Usage:
#   ./tests/bench_ssd_first.sh [model_dir] [port] [rung,rung,...]
# Env:
#   MLX_SERVE_BINARY   default ./zig-out/bin/mlx-serve
#   ARMS               "ssd,control" (default) | "ssd" | "control"
#   CTX                --ctx-size (default 1048576)
#   KV                 --kv-quant (default 8)
#   DISK               --prefix-cache-disk (default 100GB)
#   OUT                CSV path (default ~/claude-tmp/ssd-first-<date>.csv)
#
# Output: one CSV row per (arm, rung, phase) plus a summary against the bars.

set -uo pipefail

MODEL="${1:-$HOME/llm/models/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit}"
PORT="${2:-8097}"
RUNGS="${3:-4096,16384,65536,131072,262144,393216}"
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
ARMS="${ARMS:-ssd,control}"
CTX="${CTX:-1048576}"
KV="${KV:-8}"
DISK="${DISK:-100GB}"
OUT="${OUT:-$HOME/claude-tmp/ssd-first-$(date +%Y%m%d-%H%M%S).csv}"
BASE="http://127.0.0.1:$PORT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

# The judge owns 11234. Refuse rather than collide.
if [ "$PORT" = "11234" ]; then
    echo -e "${RED}REFUSING${NC}: port 11234 belongs to the judge. Pick another."
    exit 2
fi
[ -d "$MODEL" ] || { echo -e "${YELLOW}SKIP${NC}: $MODEL not found."; exit 0; }
[ -x "$BINARY" ] || { echo -e "${RED}FAIL${NC}: $BINARY missing — build ReleaseFast first."; exit 1; }

SCRATCH_HOME="$(mktemp -d)"
LOGFILE="$(mktemp)"
SERVER_PID=""
mkdir -p "$(dirname "$OUT")"

cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    wait 2>/dev/null
    rm -rf "$SCRATCH_HOME" "$LOGFILE"
}
trap cleanup EXIT

start_server() { # arm
    local arm="$1"; shift
    : > "$LOGFILE"
    local env_prefix=()
    [ "$arm" = "control" ] && env_prefix=(MLX_SERVE_PREFIX_SSD_FIRST=0)
    HOME="$SCRATCH_HOME" env "${env_prefix[@]}" "$BINARY" --model "$MODEL" --serve \
        --host 127.0.0.1 --port "$PORT" --ctx-size "$CTX" --kv-quant "$KV" \
        --prefix-cache-disk "$DISK" --log-level info "$@" > "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 900); do
        curl -s -f "$BASE/health" > /dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:"; tail -30 "$LOGFILE"; return 1; }
        sleep 1
    done
    return 1
}

stop_server() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 2; }

# A deterministic prompt of ~N tokens for chain `tag`. The chains share NO
# prefix, so one can never restore from the other's entry.
prompt_for() { # tokens tag turn
    python3 - "$1" "$2" "$3" <<'PY'
import sys
n, tag, turn = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
head = f"Session {tag}. You are auditing a ledger. Read every entry, then answer."
body = []
i = 0
while len(body) * 12 < n:
    i += 1
    body.append(f"Entry {tag}-{i}: account {i*37%9973} moved {i*131%7919} units, checksum {i*i%99991}, flag {tag}{i%7}.")
tail = f"Turn {turn}. Question: name the account in entry {tag}-42. One short sentence."
print("\n".join([head, *body, tail]))
PY
}

# Fire one request; print "ttft_ms|total_ms|cached_tokens|prompt_tokens".
fire() { # prompt max_tokens
    # The heredoc IS stdin, so the prompt must ride argv, not a pipe.
    python3 - "$BASE" "$2" "$1" <<'PY'
import json, sys, time, urllib.request
base, max_tokens, prompt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
body = json.dumps({"model": "mlx-serve", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                   "stream_options": {"include_usage": True}}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
t0 = time.monotonic(); ttft = None; cached = 0; ptok = 0
with urllib.request.urlopen(req, timeout=1800) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            ev = json.loads(payload)
        except Exception:
            continue
        if ttft is None and ev.get("choices"):
            d = ev["choices"][0].get("delta") or {}
            if d.get("content") or d.get("reasoning_content"):
                ttft = time.monotonic() - t0
        u = ev.get("usage")
        if u:
            ptok = u.get("prompt_tokens", 0)
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
total = time.monotonic() - t0
print(f"{(ttft or total)*1000:.0f}|{total*1000:.0f}|{cached}|{ptok}")
PY
}

props_active_mb() {
    curl -s "$BASE/props" 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); m=d.get('memory') or d
    print(int(m.get('active_bytes', 0))>>20)
except Exception:
    print(0)"
}

# Last flush's numbers off the persisted line, or 0|0.
last_flush() {
    # A heredoc IS stdin, so a pipe into `python3 -` would be swallowed by it.
    # The line rides argv instead.
    local line
    line=$(grep '\[disk-cache\] persisted' "$LOGFILE" | tail -1)
    python3 - "$line" <<'PY'
import re, sys
line = sys.argv[1] if len(sys.argv) > 1 else ""
mb = re.search(r"([0-9.]+) MB", line)
ms = re.search(r"(\d+)ms", line)
print((mb.group(1) if mb else "0") + "|" + (ms.group(1) if ms else "0"))
PY
}

echo "arm,rung,phase,ttft_ms,total_ms,cached_tokens,prompt_tokens,active_mb,flush_mb,flush_ms,probe_ttft_ms" > "$OUT"
echo "SSD-first harness -> $OUT"
echo "  model=$MODEL port=$PORT ctx=$CTX kv=$KV disk=$DISK arms=$ARMS"
echo "  rungs=$RUNGS"

SHORT_PROMPT="Reply with the single word: ready."

for arm in ${ARMS//,/ }; do
    echo -e "\n=== arm: $arm ==="
    start_server "$arm" || { echo -e "${RED}FAIL${NC} boot ($arm)"; exit 1; }

    # Boot-line evidence: which arm actually engaged.
    grep -E 'SSD-first budget|SSD-first chunk|background writer armed' "$LOGFILE" | sed 's/^/  /' || true
    if [ "$arm" = "ssd" ] && ! grep -q 'background writer armed' "$LOGFILE"; then
        echo -e "  ${YELLOW}WARN${NC}: SSD-first arm booted WITHOUT the background writer — arm not proven."
    fi
    if [ "$arm" = "control" ] && grep -q 'SSD-first budget' "$LOGFILE"; then
        echo -e "  ${RED}FAIL${NC}: control arm engaged SSD-first — the kill switch did not hold."
    fi

    for rung in ${RUNGS//,/ }; do
        PA=$(prompt_for "$rung" A 1)
        PB=$(prompt_for "$rung" B 1)

        # (1) COLD: chain A, first sight.
        r=$(fire "$PA" 8); IFS='|' read -r t tot c p <<< "$r"
        f=$(last_flush); IFS='|' read -r fmb fms <<< "$f"
        # (4) a short probe immediately behind it — a synchronous writer spikes here.
        pr=$(fire "$SHORT_PROMPT" 4); probe=${pr%%|*}
        echo "$arm,$rung,cold,$t,$tot,$c,$p,$(props_active_mb),$fmb,$fms,$probe" >> "$OUT"
        echo "  rung $rung cold      ttft=${t}ms cached=$c flush=${fmb}MB/${fms}ms probe=${probe}ms"

        # (2) RAM HIT: same session, nothing else has run but the tiny probe.
        r=$(fire "$PA" 8); IFS='|' read -r t tot c p <<< "$r"
        echo "$arm,$rung,ram_hit,$t,$tot,$c,$p,$(props_active_mb),,," >> "$OUT"
        echo "  rung $rung ram-hit   ttft=${t}ms cached=$c"

        # (3) DISK HIT: chain B takes the resident slot, then chain A returns.
        fire "$PB" 8 > /dev/null
        r=$(fire "$PA" 8); IFS='|' read -r t tot c p <<< "$r"
        restored=$(grep -c '\[disk-cache\] restored' "$LOGFILE")
        echo "$arm,$rung,disk_hit,$t,$tot,$c,$p,$(props_active_mb),,," >> "$OUT"
        echo "  rung $rung disk-hit  ttft=${t}ms cached=$c (restore lines so far: $restored)"

        # (2) idle residency, nothing in flight.
        sleep 3
        echo "$arm,$rung,idle,,,,,$(props_active_mb),,," >> "$OUT"
        echo "  rung $rung idle      active=$(props_active_mb)MB"
    done

    grep -cE '\[hot-cache\] SSD-first: spilled' "$LOGFILE" | xargs -I{} echo "  idle spills observed: {}"
    stop_server
done

# (5) kill -9 durability, SSD-first arm only.
echo -e "\n=== kill -9 durability ==="
start_server ssd || exit 1
BIG=$(prompt_for 131072 A 1)
fire "$BIG" 8 > /dev/null &
FIRE_PID=$!
sleep 25                        # land mid-prefill / mid-flush
kill -9 "$SERVER_PID" 2>/dev/null
wait "$FIRE_PID" 2>/dev/null
SERVER_PID=""
ENTRIES=$(find "$SCRATCH_HOME/.mlx-serve/kv-cache" -name 'meta.json' 2>/dev/null | wc -l | tr -d ' ')
TMPS=$(find "$SCRATCH_HOME/.mlx-serve/kv-cache" -name '*.tmp' 2>/dev/null | wc -l | tr -d ' ')
echo "  after SIGKILL: $ENTRIES indexed entries, $TMPS leftover .tmp files"
start_server ssd || exit 1
r=$(fire "$BIG" 8); IFS='|' read -r t tot c p <<< "$r"
echo "  restart: ttft=${t}ms cached=$c/$p"
grep -E '\[disk-cache\] (restored|scanned|dropping|salvaging)' "$LOGFILE" | sed 's/^/    /' | head
echo "ssd,131072,kill9_restart,$t,$tot,$c,$p,$(props_active_mb),,," >> "$OUT"
stop_server

echo -e "\n=== bars ==="
echo "  (evaluate against the CSV; these are the design's numbers)"
echo "  * RAM hit must not regress vs the control arm's RAM hit"
echo "  * disk hit at the top rung <= 10 s TTFT"
echo "  * idle active_mb ~= weights + ONE session (not weights + N sessions)"
echo "  * probe_ttft_ms must not spike behind a large flush (writer is off-thread)"
echo "  * kill -9: 0 leftover half-indexed entries; restart restores a chunk-aligned prefix"
echo -e "${GREEN}done${NC} -> $OUT"
