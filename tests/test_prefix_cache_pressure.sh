#!/bin/bash
# Usage:
#   PRESSURE_TEST_MODEL=<model-dir> MLX_SERVE_BINARY=<ReleaseFast-bin> \
#     bash tests/test_prefix_cache_pressure.sh [model-dir] [port]
#
# Phase 1 (default): starts mlx-serve with an isolated HOME and a small hot cache,
#   sends distinct chat requests to fill it, raises macOS system memory pressure
#   (`memory_pressure -S -l warn` first, falling back to a `vm_stat`-sized incompressible
#   allocation held by a helper process), and asserts B3's budget-step log line:
#   `[hot-cache] memory pressure worse: <warning|critical>, re-clamping the budgets`.
# Phase 2 (`MLX_SERVE_PREFIX_CACHE_PRESSURE=0`): repeats under the same pressure and
#   asserts that no pressure budget step is logged.
# Exits 0 only on pass; last line is `PRESSURE-TEST-OK` or `PRESSURE-TEST-FAIL <why>`.
# shellcheck disable=SC1091

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_lib_models.sh"
source "$SCRIPT_DIR/lib/server.sh"

MODEL="${PRESSURE_TEST_MODEL:-${DSV41_TEST_MODEL:-${1:-}}}"
PORT="${PRESSURE_TEST_PORT:-${2:-19049}}"
BINARY="${MLX_SERVE_BINARY:-${BINARY:-./zig-out/bin/mlx-serve}}"
BASE="http://127.0.0.1:$PORT"
CACHE_ENTRIES="${PRESSURE_CACHE_ENTRIES:-4}"
CACHE_MB="${PRESSURE_CACHE_MB:-256}"
CTX_SIZE="${PRESSURE_CTX_SIZE:-8192}"
FILL_REQS="${PRESSURE_FILL_REQS:-5}"
PROMPT_WORDS="${PRESSURE_PROMPT_WORDS:-900}"
ON_WAIT_SECS="${PRESSURE_ON_WAIT_SECS:-25}"
OFF_HOLD_SECS="${PRESSURE_OFF_HOLD_SECS:-14}"
STEP_PATTERN='\[hot-cache\] memory pressure worse: (warning|critical), re-clamping the budgets'

SERVER_PID=""
PRESSURE_PID=""
PRESSURE_MODE=""
LOGFILE=""
OUT=""
FINAL_LINE="PRESSURE-TEST-FAIL unexpected exit"

EXTRA_FLAGS=()
[ -z "${PRESSURE_TEST_FLAGS:-}" ] || read -ra EXTRA_FLAGS <<< "$PRESSURE_TEST_FLAGS"

fail() {
    FINAL_LINE="PRESSURE-TEST-FAIL $*"
    exit 1
}

pressure_level() {
    local lvl
    lvl=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 0)
    [[ "$lvl" =~ ^[0-9]+$ ]] && echo "$lvl" || echo 0
}

stop_pressure() {
    if [ -n "$PRESSURE_PID" ]; then
        kill -TERM "$PRESSURE_PID" 2>/dev/null || true
        sleep 0.2
        kill -KILL "$PRESSURE_PID" 2>/dev/null || true
        wait "$PRESSURE_PID" 2>/dev/null || true
        PRESSURE_PID=""
    fi
    if [ "$PRESSURE_MODE" = "memory_pressure_sim" ]; then
        memory_pressure -S -l normal >/dev/null 2>&1 || true
    fi
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        srv_stop "$SERVER_PID"
        SERVER_PID=""
    fi
}

diagnostics() {
    local log
    for log in "$OUT"/*.server.log "$OUT"/*.hog.log; do
        [ -f "$log" ] || continue
        echo "--- $log"
        tail -n 60 "$log" || true
    done
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM ERR
    [ "$FINAL_LINE" = "PRESSURE-TEST-OK" ] || [ "$rc" -ne 0 ] || rc=1
    stop_pressure
    stop_server
    if [ "$rc" -ne 0 ] && [ -n "$OUT" ] && [ -d "$OUT" ]; then
        diagnostics
        echo "artifacts: $OUT"
    elif [ -n "$OUT" ] && [ -d "$OUT" ]; then
        rm -rf "$OUT"
    fi
    echo "$FINAL_LINE"
    exit "$rc"
}

trap 'FINAL_LINE="PRESSURE-TEST-FAIL command failed at line $LINENO"' ERR
trap cleanup EXIT INT TERM

[ "$(uname)" = "Darwin" ] || fail "macOS required for kern.memorystatus_vm_pressure_level"
[ -x "$BINARY" ] || fail "binary not executable: $BINARY"

if [ -z "$MODEL" ]; then
    MODEL=$(find_fitting_model \
        mlx-community/gemma-4-e4b-it-4bit \
        mlx-community/gemma-4-e2b-it-4bit \
        lmstudio-community/Qwen3.5-2B-MLX-4bit \
        mlx-community/Qwen3-4B-4bit || true)
fi
[ -n "$MODEL" ] || fail "no model specified or found (set PRESSURE_TEST_MODEL or pass model dir)"
[ -f "$MODEL/config.json" ] || fail "missing config.json in $MODEL"

mkdir -p "$HOME/claude-tmp"
OUT=$(mktemp -d "$HOME/claude-tmp/prefix-cache-pressure.XXXXXX")

check_port_free() {
    python3 - "$PORT" <<'PY' || fail "port $PORT is already in use"
import socket, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

wait_pressure_normal() {
    local _
    for _ in $(seq 1 60); do
        [ "$(pressure_level)" -eq 1 ] && return 0
        sleep 0.5
    done
    fail "system memory pressure did not settle to normal (level=$(pressure_level))"
}

start_server() {
    local label="$1"
    local case_home="$2"
    local pressure_env="$3"
    mkdir -p "$case_home"
    LOGFILE="$OUT/$label.server.log"
    check_port_free
    if [ -n "$pressure_env" ]; then
        HOME="$case_home" MLX_SERVE_PREFIX_CACHE_PRESSURE="$pressure_env" \
            "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
            --ctx-size "$CTX_SIZE" --no-pld --no-mtp --no-drafter --no-vision --metrics \
            --prefix-cache-entries "$CACHE_ENTRIES" --prefix-cache-mem "${CACHE_MB}MB" \
            --prefix-cache-disk off --log-level info \
            ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} > "$LOGFILE" 2>&1 &
    else
        HOME="$case_home" \
            "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
            --ctx-size "$CTX_SIZE" --no-pld --no-mtp --no-drafter --no-vision --metrics \
            --prefix-cache-entries "$CACHE_ENTRIES" --prefix-cache-mem "${CACHE_MB}MB" \
            --prefix-cache-disk off --log-level info \
            ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} > "$LOGFILE" 2>&1 &
    fi
    SERVER_PID=$!
    srv_wait_health "$BASE" 600 1 "$SERVER_PID" || fail "server failed health check in $label"
}

fill_hot_cache() {
    local phase="$1"
    python3 - "$BASE" "$FILL_REQS" "$PROMPT_WORDS" "$phase" <<'PY' || fail "chat request failed while filling cache ($phase)"
import json, sys, urllib.request

base, count, words, phase = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
for idx in range(1, count + 1):
    body_words = " ".join(f"item-{phase}-{idx}-{w}-{(idx * 31 + w * 17) % 997}" for w in range(words))
    prompt = f"Distinct session {phase}-{idx}:\n{body_words}\nReply with OK."
    payload = json.dumps({
        "model": "mlx-serve",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0.0,
        "enable_thinking": False,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    msg = data.get("choices", [{}])[0].get("message", {})
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    if not text:
        raise SystemExit(f"empty completion on request {idx}")
PY
    grep -q '\[hot-cache\] resident=' "$LOGFILE" || fail "hot cache showed no resident entries in $phase"
}

try_simulated_pressure() {
    command -v memory_pressure >/dev/null 2>&1 || return 1
    memory_pressure -S -l warn >/dev/null 2>&1 &
    PRESSURE_PID=$!
    local _
    for _ in $(seq 1 10); do
        sleep 0.25
        if [ "$(pressure_level)" -ge 2 ]; then
            sleep 1
            if [ "$(pressure_level)" -ge 2 ]; then
                PRESSURE_MODE="memory_pressure_sim"
                echo "pressure_source: memory_pressure -S -l warn (level=$(pressure_level))"
                return 0
            fi
        fi
        if ! kill -0 "$PRESSURE_PID" 2>/dev/null; then
            wait "$PRESSURE_PID" 2>/dev/null || true
            PRESSURE_PID=""
            return 1
        fi
    done
    stop_pressure
    return 1
}

start_vm_stat_allocation() {
    local label="$1"
    local ready_file="$OUT/$label.ready"
    local hog_log="$OUT/$label.hog.log"
    rm -f "$ready_file"
    python3 - "$$" "$ready_file" "90" > "$hog_log" 2>&1 <<'PY' &
import ctypes, os, re, signal, subprocess, sys, time

parent_pid = int(sys.argv[1])
ready_path = sys.argv[2]
max_secs = float(sys.argv[3])

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
signal.signal(signal.SIGINT, lambda *_: os._exit(0))

libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
libc.sysctlbyname.argtypes = [
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_void_p,
    ctypes.c_size_t,
]
libc.sysctlbyname.restype = ctypes.c_int

def sysctl_u64(name, size):
    buf = ctypes.create_string_buffer(size)
    ln = ctypes.c_size_t(size)
    rc = libc.sysctlbyname(name.encode(), buf, ctypes.byref(ln), None, 0)
    if rc != 0:
        return 0
    return int.from_bytes(buf.raw[:ln.value], sys.byteorder, signed=False)

def pressure_level():
    return sysctl_u64("kern.memorystatus_vm_pressure_level", 4)

vm_out = subprocess.check_output(["vm_stat"], text=True)
m_pg = re.search(r"page size of (\d+) bytes", vm_out)
page_size = int(m_pg.group(1)) if m_pg else 16384
counts = {k.strip(): int(v) for k, v in re.findall(r"^([^:]+):\s+(\d+)\.", vm_out, re.M)}
free_pages = counts.get("Pages free", 0) + counts.get("Pages speculative", 0)
reclaim_pages = free_pages + counts.get("Pages purgeable", 0) + counts.get("File-backed pages", 0)
total_bytes = sysctl_u64("hw.memsize", 8)
free_bytes = free_pages * page_size
reclaim_bytes = reclaim_pages * page_size
cap_bytes = total_bytes - (16 << 30)

tpl_small = bytearray(os.urandom(128 << 20))
tpl_big = tpl_small * 4
chunks = []
allocated = 0
step_idx = 0
deadline = time.monotonic() + max_secs

def add_chunk(tpl):
    global allocated, step_idx
    step_idx += 1
    buf = bytearray(tpl)
    n_pages = len(buf) // page_size
    buf[0::page_size] = bytes([(step_idx * 131 + 17) & 0xFF]) * n_pages
    chunks.append(buf)
    allocated += len(buf)

while pressure_level() < 2 and allocated < cap_bytes and time.monotonic() < deadline:
    if os.getppid() != parent_pid:
        os._exit(0)
    fast_limit = max(free_bytes - (4 << 30), 0)
    use_big = allocated + len(tpl_big) <= fast_limit
    try:
        add_chunk(tpl_big if use_big else tpl_small)
    except MemoryError:
        break
    if not use_big:
        time.sleep(0.015)

lvl = pressure_level()
if lvl < 2:
    print(f"failed to reach warning pressure: level={lvl} allocated_mb={allocated >> 20}", flush=True)
    os._exit(2)

tmp_path = ready_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as fh:
    fh.write(f"level={lvl} allocated_mb={allocated >> 20} vm_stat_free_pages={free_pages} page_size={page_size}\n")
os.replace(tmp_path, ready_path)

while time.monotonic() < deadline:
    if os.getppid() != parent_pid:
        os._exit(0)
    if pressure_level() >= 4:
        print(f"critical pressure reached, releasing: allocated_mb={allocated >> 20}", flush=True)
        os._exit(3)
    if pressure_level() < 2 and allocated + len(tpl_small) <= cap_bytes:
        try:
            add_chunk(tpl_small)
        except MemoryError:
            pass
    time.sleep(0.2)

os._exit(0)
PY
    PRESSURE_PID=$!
    PRESSURE_MODE="vm_stat_alloc"
    local _
    for _ in $(seq 1 240); do
        if [ -f "$ready_file" ]; then
            local info
            info=$(cat "$ready_file")
            echo "pressure_source: vm_stat_alloc ($info)"
            return 0
        fi
        if ! kill -0 "$PRESSURE_PID" 2>/dev/null; then
            wait "$PRESSURE_PID" 2>/dev/null || true
            PRESSURE_PID=""
            fail "vm_stat allocation helper exited before reaching warning pressure ($label)"
        fi
        sleep 0.25
    done
    fail "timed out waiting for vm_stat allocation helper to raise pressure ($label)"
}

raise_pressure() {
    local label="$1"
    if try_simulated_pressure; then
        return 0
    fi
    echo "memory_pressure -S -l warn unavailable or did not move kern.memorystatus_vm_pressure_level; using vm_stat allocation"
    start_vm_stat_allocation "$label"
}

ensure_pressure_held() {
    local label="$1"
    if [ "$PRESSURE_MODE" = "memory_pressure_sim" ]; then
        if [ "$(pressure_level)" -lt 2 ]; then
            stop_pressure
            echo "simulated pressure dropped early; switching to vm_stat allocation"
            start_vm_stat_allocation "$label"
        fi
        return 0
    fi
    kill -0 "$PRESSURE_PID" 2>/dev/null || fail "$label: vm_stat allocation helper exited early"
}

log_since_matches() {
    local mark="$1"
    local pattern="$2"
    tail -n "+$((mark + 1))" "$LOGFILE" > "$OUT/slice.log"
    grep -Eq "$pattern" "$OUT/slice.log"
}

wait_for_step_line() {
    local mark="$1"
    local tries=$(( ON_WAIT_SECS * 2 ))
    local _
    for _ in $(seq 1 "$tries"); do
        if log_since_matches "$mark" "$STEP_PATTERN"; then
            return 0
        fi
        ensure_pressure_held "phase1"
        sleep 0.5
    done
    return 1
}

wait_pressure_normal
echo "== phase 1: default pressure gauge =="
start_server "phase1-on" "$OUT/on-home" ""
fill_hot_cache "on"
ON_MARK=$(wc -l < "$LOGFILE" | tr -d ' ')
raise_pressure "phase1"
if ! wait_for_step_line "$ON_MARK"; then
    fail "phase 1 (default): missing '$STEP_PATTERN' after ${ON_WAIT_SECS}s under pressure"
fi
MATCHED_STEP=$(grep -E "$STEP_PATTERN" "$OUT/slice.log" | tail -n 1)
echo "PASS phase 1: $MATCHED_STEP"
stop_pressure
wait_pressure_normal
curl -sf -m 5 "$BASE/health" >/dev/null || fail "phase 1: server unhealthy after pressure"
stop_server

echo "== phase 2: MLX_SERVE_PREFIX_CACHE_PRESSURE=0 =="
start_server "phase2-off" "$OUT/off-home" "0"
fill_hot_cache "off"
OFF_MARK=$(wc -l < "$LOGFILE" | tr -d ' ')
raise_pressure "phase2"
OFF_NEED_TICKS=$(( OFF_HOLD_SECS * 2 ))
OFF_MAX_TICKS=$(( OFF_NEED_TICKS + 24 ))
HELD_TICKS=0
for _ in $(seq 1 "$OFF_MAX_TICKS"); do
    ensure_pressure_held "phase2"
    if log_since_matches "$OFF_MARK" '\[hot-cache\] memory pressure '; then
        fail "phase 2 (MLX_SERVE_PREFIX_CACHE_PRESSURE=0): unexpected memory pressure step logged"
    fi
    if [ "$(pressure_level)" -ge 2 ]; then
        HELD_TICKS=$(( HELD_TICKS + 1 ))
        [ "$HELD_TICKS" -ge "$OFF_NEED_TICKS" ] && break
    else
        HELD_TICKS=0
    fi
    sleep 0.5
done
[ "$HELD_TICKS" -ge "$OFF_NEED_TICKS" ] || fail "phase 2: pressure did not stay >= warning for ${OFF_HOLD_SECS}s"
stop_pressure
wait_pressure_normal
curl -sf -m 5 "$BASE/health" >/dev/null || fail "phase 2: server unhealthy after pressure"
if log_since_matches "$OFF_MARK" '\[hot-cache\] (memory pressure |budget revised )'; then
    fail "phase 2 (MLX_SERVE_PREFIX_CACHE_PRESSURE=0): unexpected budget revision logged"
fi
echo "PASS phase 2: MLX_SERVE_PREFIX_CACHE_PRESSURE=0 logged no budget step"
stop_server

FINAL_LINE="PRESSURE-TEST-OK"
