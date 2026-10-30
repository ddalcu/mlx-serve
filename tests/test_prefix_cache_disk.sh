#!/bin/bash
# test_prefix_cache_disk.sh — SSD prefix-cache persistence (--prefix-cache-disk), scratch HOME (never the real
# kv-cache). [1-3] restart round trip: persist -> kill -> reboot -> `[disk-cache] restored N/M`, byte-identical temp-0
# output, >= 2x TTFT win, chunked on-disk layout. [4] a multi-turn extension appends chunks. [5] --prefix-cache-disk off
# never touches the dir. [6] --no-prefix-cache-ram persists and restores with no hot residency; entries 0 disables
# both tiers. Hybrid section (Qwen3.5-0.8B): SSM checkpoints persist and restore `(ssm@pos)` byte-identical.
# [9] image (VISION_DISK_MODEL): text + image + text, launchd-style TERM/5 s/KILL, turn 2 restores past the logged
# `media: item 0 at row N` within one chunk, RAM-first and SSD-only; another image in that slot stops at or before the
# row. [10] idle flush (IDLE_FLUSH_MODEL, a qwen3_5): two ~24k sessions both land `e<N> complete on disk` before the
# stop and restore after it. [11] SIGTERM twice mid-prefill: one `[disk-cache] shutdown persist:` summary, exit before
# the KILL, both sessions restore; at --shutdown-persist-secs 1 no half-written entry.
#
#   ./tests/test_prefix_cache_disk.sh [/path/to/model] [port]
#
# Hermetic twins: kv_disk_cache / disk_tier, prefix_cache two-tier, kv_disk_writer and armShutdownDeadline tests.

set -e
source "$(dirname "$0")/lib/server.sh"

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
PORT="${2:-8096}"
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Legacy flat layout fallback (pre-two-level model dirs).
if [ ! -d "$MODEL" ] && [ -d "$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit" ]; then
    MODEL="$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit"
fi
if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_prefix_cache_disk: $MODEL not found."
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build first."
    exit 1
fi

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

# Isolated HOME so the test never touches the user's real kv-cache.
# On the real disk: /tmp can be a small tmpfs (Linux), where the SSD tier rightly refuses to store.
mkdir -p "$HOME/claude-tmp"
SCRATCH_HOME=$(mktemp -d "$HOME/claude-tmp/prefix-disk.XXXXXX")
HYBRID_HOME=$(mktemp -d "$HOME/claude-tmp/prefix-disk-hybrid.XXXXXX")
LOGFILE=$(mktemp)
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf "$SCRATCH_HOME" "$HYBRID_HOME" "$LOGFILE"
}
trap cleanup EXIT

start_server() { # extra args...
    : > "$LOGFILE"
    # The SSD tier is OPT-IN since the off-by-default flip (ad3fd24) — the
    # suite must enable it explicitly. Callers' "$@" comes later, so section
    # 5's `--prefix-cache-disk off` still wins (last flag parses last).
    # SERVER_ULIMIT_N lowers the server's soft open-file limit.
    ( [ -n "$SERVER_ULIMIT_N" ] && ulimit -n "$SERVER_ULIMIT_N"
      HOME="$SCRATCH_HOME" exec "$BINARY" --model "$MODEL" --serve --port "$PORT" \
        --ctx-size 8192 --no-pld --log-level info --prefix-cache-disk 4GB "$@" ) > "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    srv_wait_health "$BASE" 90 1 "$SERVER_PID" && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died:"; tail -20 "$LOGFILE"; return 1; }
    return 1
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 1
}

launchd_stop() { # TERM, 5 s grace, then KILL: what `exit timeout = 5` does
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5; do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 1
}

term_stop() { # $1 seconds to wait before KILL. TERM twice, 0.5 s apart; sets STOP_SECS and KILLED
    local t0 now # wall clock: python's monotonic clock is per process on macOS
    t0=$(python3 -c 'import time; print(time.time())')
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 0.5
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    KILLED=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        now=$(python3 -c "import time; print(int(time.time() - $t0 >= $1))")
        if [ "$now" = 1 ]; then kill -KILL "$SERVER_PID" 2>/dev/null || true; KILLED=1; break; fi
        sleep 0.2
    done
    wait "$SERVER_PID" 2>/dev/null || true
    STOP_SECS=$(python3 -c "import time; print(f'{time.time() - $t0:.1f}')")
    SERVER_PID=""
    sleep 1
}

session_turn() { # $1 session label (A|B|C), $2 log lines (default 800, ~25.8k qwen3_5 tokens) -> "content"
    python3 - "$BASE" "$1" "${2:-800}" <<'PY'
import json, sys, urllib.request
base, who, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
k = {'A': 1, 'B': 7, 'C': 13}[who]
lines = [f'Session {who}: read this project log fully, then answer.']
lines += [f'Log entry {i}: subsystem {(i*k)%17} reported state change {i*31*k%997} with latency {i*7%131} ms and checksum {i*i*k%9973}.' for i in range(1, n + 1)]
lines.append('Question: which subsystem appears in log entry 42? One short sentence.')
body = json.dumps({'model': 'mlx-serve', 'messages': [{'role': 'user', 'content': '\n'.join(lines)}],
                   'max_tokens': 24, 'temperature': 0.0, 'enable_thinking': False, 'stream': False}).encode()
req = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
resp = json.load(urllib.request.urlopen(req, timeout=1200))
print(resp['choices'][0]['message']['content'].replace('\n', ' '))
PY
}

sessions_restore() { # resend sessions A and B; each must restore within one chunk of its prompt
    local who seen line restored total
    for who in A B; do
        seen=$(grep -cE '\[disk-cache\] restored [0-9]+/[0-9]+ tokens' "$LOGFILE" || true)
        session_turn "$who" > /dev/null || { echo -e "${RED}FAIL${NC} session $who after reboot failed"; FAIL=1; }
        line=$(grep -oE '\[disk-cache\] restored [0-9]+/[0-9]+ tokens' "$LOGFILE" | sed -n "$((seen + 1))p")
        restored=$(echo "$line" | grep -oE '[0-9]+/' | tr -d /)
        total=$(echo "$line" | grep -oE '/[0-9]+' | tr -d /)
        if [ -n "$restored" ] && [ "$restored" -ge $((total - 1024)) ]; then
            echo -e "${GREEN}PASS${NC} session $who restored $restored/$total after the reboot"
        else
            echo -e "${RED}FAIL${NC} session $who restored '${restored:-none}' of '${total:-?}' (want within 1024 of the prompt)"; FAIL=1
        fi
    done
}

# Deterministic ~2.5k-token prompt (well above the 512-token persist floor).
LONG_PROMPT=$(python3 -c "
lines = ['You are reviewing a project log. Read it fully, then answer.']
for i in range(1, 81):
    lines.append(f'Log entry {i}: subsystem {i%17} reported state change {i*31%997} with latency {i*7%131} ms and checksum {i*i%9973}.')
lines.append('Question: which subsystem appears in log entry 42? Answer with one short sentence.')
print('\n'.join(lines))")

fire_long() { # -> "elapsed_ms|content"
    local body
    body=$(python3 -c "
import json,sys
print(json.dumps({'model':'mlx-serve','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':24,'temperature':0.0,'stream':False}))
" "$LONG_PROMPT")
    python3 - "$BASE" "$body" <<'PY'
import json, sys, time, urllib.request
base, body = sys.argv[1], sys.argv[2]
t0 = time.monotonic()
req = urllib.request.Request(base + "/v1/chat/completions", data=body.encode(), headers={"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req, timeout=600))
ms = int((time.monotonic() - t0) * 1000)
content = resp["choices"][0]["message"]["content"].replace("|", " ").replace("\n", " ")
print(f"{ms}|{content}")
PY
}

FAIL=0
KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"

echo "== 1. cold boot + long request (persist) =="
start_server || { echo -e "${RED}FAIL${NC} server 1 failed to start"; exit 1; }
OUT1=$(fire_long)
COLD_MS="${OUT1%%|*}"
CONTENT1="${OUT1#*|}"
echo "  cold TTLT(total)=${COLD_MS}ms content='${CONTENT1:0:60}'"
sleep 1  # flush happens post-response on the inference thread
if grep -q '\[disk-cache\] persisted' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} disk persist fired"
else
    echo -e "${RED}FAIL${NC} no [disk-cache] persisted line"
    tail -20 "$LOGFILE"; FAIL=1
fi
stop_server

echo
echo "== 2. on-disk layout =="
META_COUNT=$(find "$KV_DIR" -name meta.json 2>/dev/null | wc -l | tr -d ' ')
CHUNK_COUNT=$(find "$KV_DIR" -name 'c*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
if [ "$META_COUNT" -ge 1 ] && [ "$CHUNK_COUNT" -ge 2 ]; then
    echo -e "${GREEN}PASS${NC} persisted entry on disk ($META_COUNT meta, $CHUNK_COUNT chunks)"
else
    echo -e "${RED}FAIL${NC} expected >=1 meta + >=2 chunks, got $META_COUNT/$CHUNK_COUNT"
    find "$KV_DIR" -type f 2>/dev/null | head
    FAIL=1
fi

echo
echo "== 3. restart + identical request (restore) =="
start_server || { echo -e "${RED}FAIL${NC} server 2 failed to start"; exit 1; }
OUT2=$(fire_long)
RESTART_MS="${OUT2%%|*}"
CONTENT2="${OUT2#*|}"
echo "  restart total=${RESTART_MS}ms (cold was ${COLD_MS}ms) content='${CONTENT2:0:60}'"
if grep -q '\[disk-cache\] restored .* tokens from SSD' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} SSD restore engaged: $(grep -o '\[disk-cache\] restored [^\\n]*' "$LOGFILE" | head -1)"
else
    echo -e "${RED}FAIL${NC} no [disk-cache] restored line after restart"
    tail -30 "$LOGFILE"; FAIL=1
fi
if [ "$CONTENT1" = "$CONTENT2" ]; then
    echo -e "${GREEN}PASS${NC} output byte-identical across restart restore"
else
    echo -e "${RED}FAIL${NC} output diverged: '$CONTENT1' vs '$CONTENT2'"
    FAIL=1
fi
SPEEDUP_OK=$(python3 -c "print(1 if $RESTART_MS * 2 <= $COLD_MS else 0)")
if [ "$SPEEDUP_OK" = "1" ]; then
    echo -e "${GREEN}PASS${NC} restart request >=2x faster than cold (${RESTART_MS}ms vs ${COLD_MS}ms)"
else
    echo -e "${RED}FAIL${NC} restart request not >=2x faster (${RESTART_MS}ms vs ${COLD_MS}ms)"
    FAIL=1
fi

echo
echo "== 4. multi-turn extension appends chunks =="
EXT_BODY=$(python3 -c "
import json,sys
print(json.dumps({'model':'mlx-serve','messages':[
  {'role':'user','content':sys.argv[1]},
  {'role':'assistant','content':sys.argv[2]},
  {'role':'user','content':'Now answer the same question for log entry 43.'}],
  'max_tokens':24,'temperature':0.0,'stream':False}))
" "$LONG_PROMPT" "$CONTENT2")
curl -s -X POST -H "Content-Type: application/json" -d "$EXT_BODY" "$BASE/v1/chat/completions" > /dev/null
sleep 1
PERSIST_LINES=$(grep -c '\[disk-cache\] persisted' "$LOGFILE" || true)
if [ "$PERSIST_LINES" -ge 1 ]; then
    echo -e "${GREEN}PASS${NC} extension turn persisted (append), $PERSIST_LINES persist line(s) this session"
else
    echo -e "${RED}FAIL${NC} extension turn did not persist"
    tail -20 "$LOGFILE"; FAIL=1
fi
stop_server

echo
echo "== 5. --prefix-cache-disk off leaves disk untouched =="
rm -rf "$KV_DIR"
start_server --prefix-cache-disk off || { echo -e "${RED}FAIL${NC} server 3 failed to start"; exit 1; }
OUT3=$(fire_long)
sleep 1
if [ -d "$KV_DIR" ] && [ -n "$(find "$KV_DIR" -name meta.json 2>/dev/null)" ]; then
    echo -e "${RED}FAIL${NC} kv-cache written despite --prefix-cache-disk off"
    FAIL=1
elif grep -q '\[disk-cache\]' "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} disk-cache log lines despite off"
    FAIL=1
else
    echo -e "${GREEN}PASS${NC} off switch respected"
fi
stop_server

echo
echo "== 6. SSD-only mode persists without RAM retention =="
rm -rf "$KV_DIR"
start_server --no-prefix-cache-ram || { echo -e "${RED}FAIL${NC} SSD-only server failed to start"; exit 1; }
fire_long > /dev/null
sleep 1
if grep -q 'Prefix cache: SSD ONLY' "$LOGFILE" &&
   grep -q '\[disk-cache\] e[0-9][0-9]* complete on disk' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} SSD-only mode persisted a complete prefix"
else
    echo -e "${RED}FAIL${NC} SSD-only mode did not persist a complete prefix"
    tail -30 "$LOGFILE"; FAIL=1
fi
stop_server
start_server --no-prefix-cache-ram || { echo -e "${RED}FAIL${NC} SSD-only restart failed"; exit 1; }
curl -fsS "$BASE/v1/unload-model" -H 'Content-Type: application/json' -d '{"model":"mlx-serve"}' > /dev/null
jq -nc --arg model "$MODEL" '{model:$model}' |
    curl -fsS "$BASE/v1/load-model" -H 'Content-Type: application/json' --data-binary @- > /dev/null
curl -fsS "$BASE/props" | jq -e '.settings.prefix_cache | .ram_enabled == false and .mem_bytes == 0' > /dev/null
fire_long > /dev/null
if grep -q '\[disk-cache\] restored .* tokens from SSD' "$LOGFILE" &&
   ! grep -q '\[hot-cache\] resident=' "$LOGFILE"; then
    echo -e "${GREEN}PASS${NC} SSD-only restart restored with no hot-cache residency"
else
    echo -e "${RED}FAIL${NC} SSD-only restart did not restore cleanly"
    tail -30 "$LOGFILE"; FAIL=1
fi
stop_server

echo "  -- zero entries disables all reuse even with SSD configured --"
start_server --prefix-cache-entries 0 || { echo -e "${RED}FAIL${NC} cache-off server failed"; exit 1; }
fire_long > /dev/null
if grep -q '\[disk-cache\]\|\[hot-cache\] reused' "$LOGFILE"; then
    echo -e "${RED}FAIL${NC} zero entries still reused or wrote prefixes"; FAIL=1
else
    echo -e "${GREEN}PASS${NC} zero entries disables both tiers"
fi
stop_server

echo
echo "== 7. hybrid SSM arch (Qwen 3.5 GatedDeltaNet) persists + restores SSM state =="
# Phase 3: hybrid recurrent archs persist their per-position SSM checkpoints
# beside the KV chunks and restore both across a restart. Gated on a local
# Qwen3.5-0.8B; SKIPs cleanly otherwise (the attention sections above cover the
# non-hybrid path either way).
HYBRID_MODEL="$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"
[ -d "$HYBRID_MODEL" ] || HYBRID_MODEL="$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"
if [ ! -d "$HYBRID_MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} hybrid section: Qwen3.5-0.8B-MLX-4bit not found."
else
    # Repoint the server helpers at the hybrid model + a fresh isolated HOME.
    MODEL="$HYBRID_MODEL"
    SCRATCH_HOME="$HYBRID_HOME"
    KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"

    echo "  -- cold boot + long request (persist) --"
    start_server --no-mtp || { echo -e "${RED}FAIL${NC} hybrid server failed to start"; exit 1; }
    HOUT1=$(fire_long)
    HCOLD_MS="${HOUT1%%|*}"
    HCONTENT1="${HOUT1#*|}"
    echo "  cold total=${HCOLD_MS}ms content='${HCONTENT1:0:60}'"
    # A hybrid restore is not bit-identical to cold (checkpoint replay, ~0.05
    # nats) and this prompt sits on a near-tie, so the bar is the RAM restore:
    # the SSD tier must reproduce it byte for byte.
    HOUTW=$(fire_long)
    HCONTENTW="${HOUTW#*|}"
    [ "$HCONTENT1" = "$HCONTENTW" ] || echo "  note: RAM-warm output differs from cold (hybrid restore class): '${HCONTENTW:0:60}'"
    sleep 1
    if grep -q '\[disk-cache\] persisted' "$LOGFILE"; then
        # A hybrid persist reports SSM checkpoints in the count.
        echo -e "${GREEN}PASS${NC} hybrid persist fired: $(grep -o '\[disk-cache\] persisted[^;]*' "$LOGFILE" | tail -1)"
    else
        echo -e "${RED}FAIL${NC} hybrid: no [disk-cache] persisted line"
        tail -20 "$LOGFILE"; FAIL=1
    fi
    stop_server

    echo "  -- restart + identical request (restore SSM@pos) --"
    start_server --no-mtp || { echo -e "${RED}FAIL${NC} hybrid server 2 failed to start"; exit 1; }
    HOUT2=$(fire_long)
    HRESTART_MS="${HOUT2%%|*}"
    HCONTENT2="${HOUT2#*|}"
    echo "  restart total=${HRESTART_MS}ms (cold was ${HCOLD_MS}ms) content='${HCONTENT2:0:60}'"
    if grep -qE '\[disk-cache\] restored .* from SSD .*ssm@[0-9]+' "$LOGFILE"; then
        echo -e "${GREEN}PASS${NC} hybrid SSD restore engaged: $(grep -oE '\[disk-cache\] restored [^(]*\([^)]*ssm@[0-9]+[^)]*\)' "$LOGFILE" | head -1)"
    else
        echo -e "${RED}FAIL${NC} hybrid: no '[disk-cache] restored … ssm@…' line after restart"
        tail -30 "$LOGFILE"; FAIL=1
    fi
    if [ "$HCONTENTW" = "$HCONTENT2" ]; then
        echo -e "${GREEN}PASS${NC} hybrid SSD restore matches the RAM restore byte for byte"
    else
        echo -e "${RED}FAIL${NC} hybrid SSD restore diverged from the RAM restore: '$HCONTENTW' vs '$HCONTENT2'"
        FAIL=1
    fi
    HSPEEDUP_OK=$(python3 -c "print(1 if $HRESTART_MS * 2 <= $HCOLD_MS else 0)")
    if [ "$HSPEEDUP_OK" = "1" ]; then
        echo -e "${GREEN}PASS${NC} hybrid restart request >=2x faster than cold (${HRESTART_MS}ms vs ${HCOLD_MS}ms)"
    else
        echo -e "${YELLOW}WARN${NC} hybrid restart not >=2x faster (${HRESTART_MS}ms vs ${HCOLD_MS}ms) — small model, TTFT dominated by fixed costs"
    fi
    stop_server
fi

echo
echo "== 8. a restore wider than the fd limit, and a failed restore's fallback =="
# A restore reads one chunk file per 1024 tokens. It must fit under a soft
# RLIMIT_NOFILE smaller than its chunk count, and an unreadable chunk must
# fall back to a cold prefill that answers 200.
MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit}"
FD_HOME=$(mktemp -d "$HOME/claude-tmp/prefix-disk-fd.XXXXXX")
SCRATCH_HOME="$FD_HOME"
KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"
LONG_PROMPT=$(python3 -c "
lines = ['You are reviewing a project log. Read it fully, then answer.']
for i in range(1, 1201):
    lines.append(f'Log entry {i}: subsystem {i%17} reported state change {i*31%997} with latency {i*7%131} ms and checksum {i*i%9973}.')
lines.append('Question: which subsystem appears in log entry 42? Answer with one short sentence.')
print('\n'.join(lines))")
start_server --ctx-size 65536 || { echo -e "${RED}FAIL${NC} fd server failed to start"; exit 1; }
# One commit flushes at most 512 MB; each repeat extends the entry.
for _ in 1 2 3 4 5 6; do
    fire_long > /dev/null
    sleep 1
    [ "$(find "$KV_DIR" -name 'c*.safetensors' 2>/dev/null | wc -l)" -ge 30 ] && break
done
TOP_FD=$(lsof -a -p "$SERVER_PID" -d '0-65535' -F f 2>/dev/null | sed -n 's/^f//p' | sort -n | tail -1)
stop_server
FD_CHUNKS=$(find "$KV_DIR" -name 'c*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
FD_LIMIT=$((TOP_FD + 16))
echo "  persisted $FD_CHUNKS chunks; serving top fd $TOP_FD; restoring under ulimit -n $FD_LIMIT"
if [ "$FD_CHUNKS" -lt 30 ]; then
    echo -e "${RED}FAIL${NC} need 30+ chunks to exceed the fd headroom, got $FD_CHUNKS"
    FAIL=1
else
    SERVER_ULIMIT_N="$FD_LIMIT" start_server --ctx-size 65536 || { echo -e "${RED}FAIL${NC} fd server 2 failed to start"; exit 1; }
    if fire_long > /dev/null && grep -q '\[disk-cache\] restored .* tokens from SSD' "$LOGFILE"; then
        echo -e "${GREEN}PASS${NC} restore under the fd limit: $(grep -oE '\[disk-cache\] restored .* in [0-9]+ms' "$LOGFILE" | head -1)"
    else
        echo -e "${RED}FAIL${NC} restore under ulimit -n $FD_LIMIT did not engage"
        grep -E 'Failed to open|restore failed|prefill failed' "$LOGFILE" | head -5; FAIL=1
    fi
    stop_server

    # After boot: the startup scan salvages an entry whose chunk is already unreadable.
    start_server --ctx-size 65536 || { echo -e "${RED}FAIL${NC} fd server 3 failed to start"; exit 1; }
    chmod 000 "$(find "$KV_DIR" -name 'c000002.safetensors' | head -1)"
    if fire_long > /dev/null && grep -q 'restore failed: .* falling back' "$LOGFILE" && ! grep -q 'prefill failed' "$LOGFILE"; then
        echo -e "${GREEN}PASS${NC} unreadable chunk: restore failed, cold prefill answered"
    else
        echo -e "${RED}FAIL${NC} unreadable chunk did not fall back to a working cold prefill"
        grep -E 'disk-cache|hot-cache|prefill failed|GenerationFailed' "$LOGFILE" | head -8; FAIL=1
    fi
    stop_server
fi
chmod -R u+rw "$FD_HOME" 2>/dev/null; rm -rf "$FD_HOME"

echo
echo "== 9. an image session persists and restores past its first image =="
# Turn 1 is ~3k tokens of text, then a message with an image and ~6k more. The server is stopped the
# way launchd stops it (TERM, 5 s, KILL), rebooted, and turn 2 is sent. The restore
# must land past the image's first row (logged as `media: item 0 at row N`) and
# within one chunk of turn 2's prompt. Turn 2 with ANOTHER image in turn 1's place
# must stop at or before that row: its rows are keyed on the pixels.
# Arms: default (RAM-first), --no-prefix-cache-ram (SSD-first + write-through), and
# the swapped image. Needs a vision model; SKIPs otherwise.
VISION_MODEL="${VISION_DISK_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit}"
IMG_A="tests/fixtures/street-name-signs.jpg"
IMG_B="tests/fixtures/house.jpeg"
if [ ! -f "$VISION_MODEL/config.json" ] || ! grep -q vision_config "$VISION_MODEL/config.json" || [ ! -f "$IMG_A" ] || [ ! -f "$IMG_B" ]; then
    echo -e "${YELLOW}SKIP${NC} image section: no vision model at $VISION_MODEL (set VISION_DISK_MODEL) or a fixture is missing."
else
    MODEL="$VISION_MODEL"

    image_turn() { # $1 turn (1|2), $2 image for turn 1's slot -> "content"
        python3 - "$BASE" "$1" "$2" <<'PY'
import base64, json, sys, urllib.request
base, turn, img = sys.argv[1], sys.argv[2], sys.argv[3]
def log(n0, n1):
    return '\n'.join(f'Log entry {i}: subsystem {i%17} reported state change {i*31%997} with latency {i*7%131} ms and checksum {i*i%9973}.' for i in range(n0, n1))
mime = 'image/png' if img.endswith('.png') else 'image/jpeg'
url = f'data:{mime};base64,' + base64.b64encode(open(img, 'rb').read()).decode()
# A template may render a message's images before its text, so the leading text is its own turn.
msgs = [{'role': 'user', 'content': 'Read this project log; a photo and the rest follow.\n' + log(1, 100)},
        {'role': 'assistant', 'content': 'Understood. Send the photo and the rest of the log.'},
        {'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': url}},
            {'type': 'text', 'text': log(100, 300) + '\nQuestion: which subsystem appears in log entry 42? One short sentence.'},
        ]}]
if turn == '2':
    msgs += [{'role': 'assistant', 'content': 'Log entry 42 names subsystem 8.'},
             {'role': 'user', 'content': 'Now the same question for log entry 43.'}]
body = json.dumps({'model': 'mlx-serve', 'messages': msgs, 'max_tokens': 24, 'temperature': 0.0,
                   'enable_thinking': False, 'stream': False}).encode()
req = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
resp = json.load(urllib.request.urlopen(req, timeout=900))
print(resp['choices'][0]['message']['content'].replace('\n', ' '))
PY
    }

    image_arm() { # $1 label, $2 turn-2 image for turn 1's slot, $3 same|other, extra server args...
        local label="$1" img2="$2" want="$3"
        shift 3
        SCRATCH_HOME=$(mktemp -d "$HOME/claude-tmp/prefix-disk-image.XXXXXX")
        KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"
        echo "  -- $label --"
        start_server --ctx-size 32768 --log-level debug "$@" || { echo -e "${RED}FAIL${NC} $label: server failed to start"; FAIL=1; return; }
        image_turn 1 "$IMG_A" > /dev/null || { echo -e "${RED}FAIL${NC} $label: turn 1 failed"; FAIL=1; launchd_stop; return; }
        local start_row
        start_row=$(grep -oE 'media: item 0 at row [0-9]+' "$LOGFILE" | head -1 | grep -oE '[0-9]+$')
        sleep 2 # the finish-path flush runs after the response
        launchd_stop
        start_server --ctx-size 32768 --log-level debug "$@" || { echo -e "${RED}FAIL${NC} $label: reboot failed"; FAIL=1; return; }
        local reply
        reply=$(image_turn 2 "$img2") || { echo -e "${RED}FAIL${NC} $label: turn 2 failed"; FAIL=1; stop_server; return; }
        local line restored total
        line=$(grep -oE '\[disk-cache\] restored [0-9]+/[0-9]+ tokens' "$LOGFILE" | head -1)
        restored=$(echo "$line" | grep -oE '[0-9]+/' | tr -d /)
        total=$(echo "$line" | grep -oE '/[0-9]+' | tr -d /)
        echo "  image row ${start_row:-?}, ${line:-no restore line}, reply '${reply:0:50}'"
        if [ -z "$start_row" ]; then
            echo -e "${RED}FAIL${NC} $label: no 'media: item 0 at row' line"; FAIL=1
        elif [ "$want" = same ] && [ -n "$restored" ] && [ "$restored" -gt "$start_row" ] && [ "$restored" -ge $((total - 1024)) ]; then
            echo -e "${GREEN}PASS${NC} $label: restored $restored/$total, past the image at row $start_row"
        elif [ "$want" = other ] && { [ -z "$restored" ] || [ "$restored" -le "$start_row" ]; }; then
            echo -e "${GREEN}PASS${NC} $label: another image stopped the restore at ${restored:-0} (image row $start_row)"
        else
            echo -e "${RED}FAIL${NC} $label: restored '${restored:-none}' of '${total:-?}' against image row $start_row (want $want)"
            grep -E '\[disk-cache\]|\[hot-cache\]' "$LOGFILE" | tail -12; FAIL=1
        fi
        stop_server
        rm -rf "$SCRATCH_HOME"
    }

    image_arm "RAM-first" "$IMG_A" same
    image_arm "SSD-first (--no-prefix-cache-ram)" "$IMG_A" same --no-prefix-cache-ram
    image_arm "another image in turn 1's slot" "$IMG_B" other
fi

echo
echo "== 10. every finished session lands on disk while idle, not only the newest =="
# Two ~24k-token sessions run together, so no idle gap separates their finishes, and each is
# several 512 MB slices. The idle loop must then land both whole (it used to finish only the
# newest), the RAM-first boot must arm the background writer, and after a launchd-style stop both
# must restore within one chunk of their prompts. Needs a hybrid qwen3_5 model; SKIPs otherwise.
IDLE_MODEL="${IDLE_FLUSH_MODEL:-$VISION_MODEL}"
if [ ! -f "$IDLE_MODEL/config.json" ] || ! grep -q '"qwen3_5' "$IDLE_MODEL/config.json"; then
    echo -e "${YELLOW}SKIP${NC} idle-flush section: no qwen3_5 model at $IDLE_MODEL (set IDLE_FLUSH_MODEL)."
else
    MODEL="$IDLE_MODEL"
    SCRATCH_HOME=$(mktemp -d "$HOME/claude-tmp/prefix-disk-idle.XXXXXX")
    KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"

    start_server --ctx-size 65536 --no-mtp || { echo -e "${RED}FAIL${NC} idle-flush server failed to start"; FAIL=1; }
    if [ -n "$SERVER_PID" ]; then
        if grep -q '\[disk-cache\] background writer armed' "$LOGFILE"; then
            echo -e "${GREEN}PASS${NC} RAM-first boot armed the background writer"
        else
            echo -e "${RED}FAIL${NC} RAM-first boot did not arm the background writer"; FAIL=1
        fi
        session_turn A > /dev/null &
        PID_A=$!
        sleep 2
        session_turn B > /dev/null &
        PID_B=$!
        wait "$PID_A" || { echo -e "${RED}FAIL${NC} session A failed"; FAIL=1; }
        wait "$PID_B" || { echo -e "${RED}FAIL${NC} session B failed"; FAIL=1; }
        # Both whole: two distinct entries logged complete at 20k+ tokens. Red waits out the timeout.
        WHOLE=0
        for _ in $(seq 1 240); do
            WHOLE=$(grep -oE '\[disk-cache\] e[0-9]+ complete on disk: [0-9]+ tokens' "$LOGFILE" |
                awk '$6 >= 20000 {print $2}' | sort -u | wc -l | tr -d ' ')
            [ "$WHOLE" -ge 2 ] && break
            sleep 1
        done
        echo "  sessions whole on disk before the stop: $WHOLE of 2"
        grep -E '\[disk-cache\] (persisted|e[0-9]+ complete)' "$LOGFILE" | sed 's/^/    /' | tail -14
        launchd_stop
        if start_server --ctx-size 65536 --no-mtp; then
            sessions_restore
            stop_server
        else
            echo -e "${RED}FAIL${NC} idle-flush reboot failed"; FAIL=1
        fi
    fi
    rm -rf "$SCRATCH_HOME"
fi

echo
echo "== 11. SIGTERM lands the slices the idle flush has not, then exits by its deadline =="
# Two ~25.8k-token sessions finish while a third, longer one is still prefilling, so the idle
# flush never runs and both still owe several 512 MB slices when SIGTERM (twice, 0.5 s apart)
# follows the second reply. The server must log one `shutdown persist:` summary and exit before
# the KILL; at the default deadline both must restore within one chunk after the reboot. At
# --shutdown-persist-secs 1 the exit stays prompt and the reboot scan finds no half-written entry.
# Needs a hybrid qwen3_5 model; SKIPs otherwise.
if [ ! -f "$IDLE_MODEL/config.json" ] || ! grep -q '"qwen3_5' "$IDLE_MODEL/config.json"; then
    echo -e "${YELLOW}SKIP${NC} shutdown-flush section: no qwen3_5 model at $IDLE_MODEL (set IDLE_FLUSH_MODEL)."
else
    MODEL="$IDLE_MODEL"
    shutdown_arm() { # $1 label, $2 --shutdown-persist-secs, $3 seconds before KILL, $4 restore|scan
        SCRATCH_HOME=$(mktemp -d "$HOME/claude-tmp/prefix-disk-term.XXXXXX")
        KV_DIR="$SCRATCH_HOME/.mlx-serve/kv-cache"
        echo "  -- $1"
        # 2048-token chunks: a cancelled prefill stops at the next chunk, and exit waits for it.
        # RAM and disk hold all three sessions, so no eviction stands in for the flush.
        local flags=(--ctx-size 65536 --no-mtp --prefill-chunk 2048 --prefix-cache-mem 16GB --prefix-cache-disk 16GB --shutdown-persist-secs "$2")
        if ! start_server "${flags[@]}"; then
            echo -e "${RED}FAIL${NC} $1: server failed to start"; FAIL=1; rm -rf "$SCRATCH_HOME"; return
        fi
        session_turn A > /dev/null &
        local pid_a=$!
        sleep 2
        session_turn B > /dev/null &
        local pid_b=$!
        sleep 2
        session_turn C 1500 > /dev/null 2>&1 & # ~48k tokens: still prefilling when B ends
        local pid_c=$!
        wait "$pid_a" || { echo -e "${RED}FAIL${NC} $1: session A failed"; FAIL=1; }
        wait "$pid_b" || { echo -e "${RED}FAIL${NC} $1: session B failed"; FAIL=1; }
        local whole
        whole=$(grep -oE '\[disk-cache\] e[0-9]+ complete on disk: [0-9]+ tokens' "$LOGFILE" | awk '$6 >= 20000' | wc -l | tr -d ' ')
        echo "  sessions whole on disk at SIGTERM: $whole of 2"
        term_stop "$3"
        wait "$pid_c" 2>/dev/null || true
        grep -E '\[disk-cache\] (persisted|e[0-9]+ complete)' "$LOGFILE" | sed 's/^/    /' | tail -8
        local summary
        summary=$(grep -E '\[disk-cache\] shutdown persist:' "$LOGFILE" || true)
        if [ -n "$summary" ]; then
            echo -e "${GREEN}PASS${NC} $1: summary logged:$(echo "$summary" | sed 's/.*shutdown persist://')"
            # The idle flush can finish both sessions before the signal; then SIGTERM wrote nothing.
            if [ "$whole" = 2 ] && echo "$summary" | grep -qE 'persist: 0 written'; then
                echo -e "${YELLOW}WARN${NC} $1: both sessions were whole before SIGTERM, so its write path did not run"
            fi
        else
            echo -e "${RED}FAIL${NC} $1: no 'shutdown persist:' summary line"; FAIL=1
        fi
        if [ "$KILLED" = 0 ]; then
            echo -e "${GREEN}PASS${NC} $1: exited ${STOP_SECS}s after SIGTERM"
        else
            echo -e "${RED}FAIL${NC} $1: still running ${STOP_SECS}s after SIGTERM, killed"; FAIL=1
        fi
        if ! start_server "${flags[@]}"; then
            echo -e "${RED}FAIL${NC} $1: reboot failed"; FAIL=1; rm -rf "$SCRATCH_HOME"; return
        fi
        if [ "$4" = restore ]; then
            sessions_restore
        else
            local scanned
            scanned=$(grep -oE '\[disk-cache\] scanned [0-9]+ persisted entries' "$LOGFILE" | grep -oE '[0-9]+' || true)
            if [ "${scanned:-0}" -ge 1 ] && ! grep -q '\[disk-cache\] dropping incomplete entry' "$LOGFILE"; then
                echo -e "${GREEN}PASS${NC} $1: the reboot scanned $scanned entries, none half-written"
            else
                echo -e "${RED}FAIL${NC} $1: reboot scan found '${scanned:-none}' entries or dropped an incomplete one"; FAIL=1
            fi
        fi
        stop_server
        rm -rf "$SCRATCH_HOME"
    }
    shutdown_arm "default deadline" 45 60 restore
    shutdown_arm "1 s deadline" 1 15 scan
fi

echo
if [ "$FAIL" = "0" ]; then
    echo -e "${GREEN}ALL PASS${NC} test_prefix_cache_disk"
else
    echo -e "${RED}FAILURES${NC} test_prefix_cache_disk"
fi
exit $FAIL
