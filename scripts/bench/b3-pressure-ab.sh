#!/bin/bash
# Usage:
#   scripts/bench/b3-pressure-ab.sh <base-bin> <head-bin> <model-dir> [out-dir]
#   scripts/bench/b3-pressure-ab.sh <bin> <model-dir> [out-dir]
#
# Runs ABBA (base, head, head, base) on one macOS box with a fresh server load per visit.
# With two binaries, compares <base-bin> vs <head-bin> at default env.
# With one binary, compares MLX_SERVE_PREFIX_CACHE_PRESSURE=0 (base) vs default on (head).
# Each visit seeds N cached sessions, lets the SSD tier settle, steps a `vm_stat`-sized
# memory hog to warning (B3_CRITICAL=1 then goes on to critical), releases pressure, and resumes every session.
# The hog never allocates past physical RAM minus 16 GiB.
# Outputs per-visit JSON, results.tsv, and summary.md (budget steps, resident bytes,
# swap used, restore time, and tokens re-prefilled per visit and per arm).

set -euo pipefail

usage() {
    sed -n '2,12p' "$0"
}

[ "${1:-}" != "-h" ] && [ "${1:-}" != "--help" ] || { usage; exit 0; }
[ "$(uname)" = "Darwin" ] || { echo "error: macOS required" >&2; exit 1; }

SINGLE_BIN_MODE=0
BASE_BIN=""
HEAD_BIN=""
MODEL=""
OUT_DIR=""

if [ "$#" -ge 3 ] && [ -d "$3" ]; then
    BASE_BIN="$1"
    HEAD_BIN="$2"
    MODEL="$3"
    OUT_DIR="${4:-}"
elif [ "$#" -ge 2 ] && [ -d "$2" ]; then
    BASE_BIN="$1"
    HEAD_BIN="$1"
    MODEL="$2"
    OUT_DIR="${3:-}"
    SINGLE_BIN_MODE=1
elif [ "$#" -ge 1 ] && [ -d "$1" ]; then
    MODEL="$1"
    OUT_DIR="${2:-}"
    if [ -n "${B3_BASE_BIN:-}" ] && [ -n "${B3_HEAD_BIN:-}" ]; then
        BASE_BIN="$B3_BASE_BIN"
        HEAD_BIN="$B3_HEAD_BIN"
    else
        BASE_BIN="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
        HEAD_BIN="$BASE_BIN"
        SINGLE_BIN_MODE=1
    fi
else
    usage >&2
    exit 2
fi

[ -x "$BASE_BIN" ] || { echo "error: base binary not executable: $BASE_BIN" >&2; exit 1; }
[ -x "$HEAD_BIN" ] || { echo "error: head binary not executable: $HEAD_BIN" >&2; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "error: config.json not found in $MODEL" >&2; exit 1; }

if [ -z "$OUT_DIR" ]; then
    mkdir -p "$HOME/claude-tmp"
    OUT_DIR="$HOME/claude-tmp/b3-pressure-ab-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT_DIR"

PORT="${B3_PORT:-19050}"
BASE_URL="http://127.0.0.1:$PORT"
SESSIONS="${B3_SESSIONS:-4}"
SESSION_LINES="${B3_SESSION_LINES:-240}"
CTX_SIZE="${B3_CTX_SIZE:-16384}"
CACHE_ENTRIES="${B3_CACHE_ENTRIES:-8}"
CACHE_MEM="${B3_CACHE_MEM:-0}"
CACHE_DISK="${B3_CACHE_DISK:-32GB}"
WARN_HOLD_SECS="${B3_WARN_HOLD_SECS:-12}"
CRIT_HOLD_SECS="${B3_CRIT_HOLD_SECS:-5}"

SERVER_PID=""
HOG_PID_FILE="$OUT_DIR/hog.pid"
EXTRA_FLAGS=()
[ -z "${B3_EXTRA_FLAGS:-}" ] || read -ra EXTRA_FLAGS <<< "$B3_EXTRA_FLAGS"

stop_hog() {
    if [ -f "$HOG_PID_FILE" ]; then
        local hpid
        hpid=$(cat "$HOG_PID_FILE" 2>/dev/null || true)
        if [ -n "$hpid" ]; then
            kill -TERM "$hpid" 2>/dev/null || true
            sleep 0.2
            kill -KILL "$hpid" 2>/dev/null || true
        fi
        rm -f "$HOG_PID_FILE"
    fi
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        local _
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

cleanup() {
    stop_hog
    stop_server
}
trap cleanup EXIT INT TERM

wait_health() {
    local pid="$1"
    local _
    for _ in $(seq 1 600); do
        curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

wait_pressure_normal() {
    local _ lvl
    for _ in $(seq 1 120); do
        lvl=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 0)
        [ "$lvl" = "1" ] && return 0
        sleep 0.5
    done
    echo "error: memory pressure stayed above normal (level=${lvl:-unknown})" >&2
    exit 1
}

check_port_free() {
    python3 - "$PORT" <<'PY'
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

bin_sha256() {
    shasum -a 256 "$1" | awk '{print $1}'
}

bin_version() {
    "$1" --version 2>&1 | head -n 1 || echo "unknown"
}

BASE_SHA=$(bin_sha256 "$BASE_BIN")
HEAD_SHA=$(bin_sha256 "$HEAD_BIN")
BASE_VER=$(bin_version "$BASE_BIN")
HEAD_VER=$(bin_version "$HEAD_BIN")

cat > "$OUT_DIR/provenance.txt" <<EOF
date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
host=$(hostname)
model=$MODEL
mode=$([ "$SINGLE_BIN_MODE" -eq 1 ] && echo "single-bin-env-toggle" || echo "two-binaries")
base_bin=$BASE_BIN
base_sha256=$BASE_SHA
base_version=$BASE_VER
head_bin=$HEAD_BIN
head_sha256=$HEAD_SHA
head_version=$HEAD_VER
sessions=$SESSIONS
session_lines=$SESSION_LINES
ctx_size=$CTX_SIZE
cache_entries=$CACHE_ENTRIES
cache_mem=$CACHE_MEM
cache_disk=$CACHE_DISK
warn_hold_secs=$WARN_HOLD_SECS
crit_hold_secs=$CRIT_HOLD_SECS
EOF
cat "$OUT_DIR/provenance.txt"

run_visit() {
    local visit_idx="$1"
    local arm="$2"
    local visit_tag="visit-${visit_idx}-${arm}"
    local visit_dir="$OUT_DIR/$visit_tag"
    local visit_home="$visit_dir/home"
    local server_log="$visit_dir/server.log"
    local bin="$HEAD_BIN"
    local pressure_env=""

    if [ "$arm" = "base" ]; then
        bin="$BASE_BIN"
        [ "$SINGLE_BIN_MODE" -eq 1 ] && pressure_env="0"
    fi

    mkdir -p "$visit_home"
    wait_pressure_normal
    check_port_free || { echo "error: port $PORT already in use before $visit_tag" >&2; exit 1; }

    echo "== $visit_tag (bin=$bin env_pressure=${pressure_env:-default}) =="
    if [ -n "$pressure_env" ]; then
        HOME="$visit_home" MLX_SERVE_PREFIX_CACHE_PRESSURE="$pressure_env" \
            "$bin" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
            --ctx-size "$CTX_SIZE" --no-pld --no-mtp --no-drafter --no-vision --metrics \
            --prefix-cache-entries "$CACHE_ENTRIES" --prefix-cache-mem "$CACHE_MEM" \
            --prefix-cache-disk "$CACHE_DISK" --shutdown-persist-secs 0 --log-level info \
            ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} > "$server_log" 2>&1 &
    else
        HOME="$visit_home" \
            "$bin" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
            --ctx-size "$CTX_SIZE" --no-pld --no-mtp --no-drafter --no-vision --metrics \
            --prefix-cache-entries "$CACHE_ENTRIES" --prefix-cache-mem "$CACHE_MEM" \
            --prefix-cache-disk "$CACHE_DISK" --shutdown-persist-secs 0 --log-level info \
            ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} > "$server_log" 2>&1 &
    fi
    SERVER_PID=$!
    if ! wait_health "$SERVER_PID"; then
        tail -n 40 "$server_log" >&2 || true
        echo "error: server failed health check on $visit_tag" >&2
        exit 1
    fi

    python3 - \
        "$BASE_URL" "$SERVER_PID" "$server_log" "$visit_dir/metrics.json" \
        "$visit_idx" "$arm" "$SESSIONS" "$SESSION_LINES" \
        "$WARN_HOLD_SECS" "$CRIT_HOLD_SECS" "$visit_dir" "$HOG_PID_FILE" <<'PY'
import ctypes, json, os, pathlib, re, subprocess, sys, time, urllib.request

(
    base_url,
    srv_pid_s,
    log_path,
    out_json,
    visit_s,
    arm,
    n_sessions_s,
    n_lines_s,
    warn_secs_s,
    crit_secs_s,
    visit_dir_s,
    hog_pid_file_s,
) = sys.argv[1:13]
srv_pid = int(srv_pid_s)
visit = int(visit_s)
n_sessions = int(n_sessions_s)
n_lines = int(n_lines_s)
warn_secs = float(warn_secs_s)
crit_secs = float(crit_secs_s)
crit_level = 4 if os.environ.get("B3_CRITICAL") == "1" else 2
visit_dir = pathlib.Path(visit_dir_s)
hog_pid_file = pathlib.Path(hog_pid_file_s)

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
    if libc.sysctlbyname(name.encode(), buf, ctypes.byref(ln), None, 0) != 0:
        return 0
    return int.from_bytes(buf.raw[:ln.value], sys.byteorder, signed=False)

def pressure_level():
    return sysctl_u64("kern.memorystatus_vm_pressure_level", 4)

def vm_stat_snapshot():
    txt = subprocess.check_output(["vm_stat"], text=True)
    m_pg = re.search(r"page size of (\d+) bytes", txt)
    pg = int(m_pg.group(1)) if m_pg else 16384
    vals = {k.strip(): int(v) for k, v in re.findall(r"^([^:]+):\s+(\d+)\.", txt, re.M)}
    return pg, vals

def swap_used_bytes():
    txt = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    m = re.search(r"used\s*=\s*([0-9.]+)\s*([KMGTP]?)", txt)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2).upper()
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}.get(unit, 1024**2)
    return int(val * scale)

def rss_bytes(pid):
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
    return int(out) * 1024 if out else 0

def get_props():
    with urllib.request.urlopen(base_url + "/props", timeout=30) as resp:
        return json.load(resp)

def chat_turn(messages, max_tokens=16):
    body = json.dumps({
        "model": "mlx-serve",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "enable_thinking": False,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
    wall_ms = (time.monotonic() - t0) * 1000.0
    msg = data.get("choices", [{}])[0].get("message", {})
    reply = msg.get("content") or msg.get("reasoning_content") or ""
    usage = data.get("usage", {})
    prompt_toks = int(usage.get("prompt_tokens", 0))
    cached_toks = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    timings = data.get("timings") or {}
    prompt_ms = float(timings.get("prompt_ms", wall_ms))
    return reply, prompt_toks, cached_toks, prompt_ms, wall_ms

HOG_CODE = r"""
import ctypes, os, pathlib, re, signal, subprocess, sys, time

parent_pid = int(sys.argv[1])
stage_path = pathlib.Path(sys.argv[2])
status_path = pathlib.Path(sys.argv[3])
max_secs = float(sys.argv[4])

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
    if libc.sysctlbyname(name.encode(), buf, ctypes.byref(ln), None, 0) != 0:
        return 0
    return int.from_bytes(buf.raw[:ln.value], sys.byteorder, signed=False)

def pressure_level():
    return sysctl_u64("kern.memorystatus_vm_pressure_level", 4)

txt = subprocess.check_output(["vm_stat"], text=True)
m_pg = re.search(r"page size of (\d+) bytes", txt)
pg = int(m_pg.group(1)) if m_pg else 16384
vals = {k.strip(): int(v) for k, v in re.findall(r"^([^:]+):\s+(\d+)\.", txt, re.M)}
free_pages = vals.get("Pages free", 0) + vals.get("Pages speculative", 0)
reclaim_pages = free_pages + vals.get("Pages purgeable", 0) + vals.get("File-backed pages", 0)
total_bytes = sysctl_u64("hw.memsize", 8)
free_bytes = free_pages * pg
cap_bytes = total_bytes - (16 << 30)

tpl_64 = bytearray(os.urandom(64 << 20))
tpl_128 = tpl_64 * 2
tpl_512 = tpl_128 * 4
chunks = []
allocated = 0
seq = 0
deadline = time.monotonic() + max_secs

def push(tpl):
    global allocated, seq
    seq += 1
    buf = bytearray(tpl)
    buf[0::pg] = bytes([(seq * 131 + 19) & 0xFF]) * (len(buf) // pg)
    chunks.append(buf)
    allocated += len(buf)
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(str(allocated >> 20), encoding="utf-8")
    os.replace(tmp, status_path)

while time.monotonic() < deadline:
    if os.getppid() != parent_pid:
        os._exit(0)
    try:
        target = int(stage_path.read_text(encoding="utf-8").strip())
    except OSError:
        target = 0
    if target <= 0:
        os._exit(0)
    lvl = pressure_level()
    if lvl < target and allocated < cap_bytes:
        fast_limit = max(free_bytes - (4 << 30), 0)
        if target == 2 and allocated + len(tpl_512) <= fast_limit:
            tpl = tpl_512
            pause = 0.0
        elif target == 2:
            tpl = tpl_128
            pause = 0.015
        else:
            tpl = tpl_64
            pause = 0.02
        try:
            push(tpl)
        except MemoryError:
            time.sleep(0.2)
        if pause:
            time.sleep(pause)
    else:
        time.sleep(0.15)

os._exit(0)
"""

conversations = []
seed_prompt_tokens = 0
for s_idx in range(1, n_sessions + 1):
    sys_msg = f"You are a concise analyst for session {visit}-{s_idx}. Answer in one short sentence."
    lines = [f"Record {s_idx}.{row}: module={(row * s_idx) % 23} cost={(row * 37 + s_idx) % 991}ms tag={row * row % 9973}" for row in range(1, n_lines + 1)]
    user1 = "\n".join(lines) + f"\nSummarize record {s_idx}.10 briefly."
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user1}]
    reply1, p_toks, _, _, _ = chat_turn(msgs, max_tokens=16)
    seed_prompt_tokens += p_toks
    conversations.append((msgs, reply1))

time.sleep(4.0)

props_pre = get_props()
mem_pre = props_pre.get("memory", {})
kv_bytes_pre = int(mem_pre.get("kv_cache_bytes", 0))
active_bytes_pre = int(mem_pre.get("active_bytes", 0))
budget_bytes_pre = int(((props_pre.get("settings") or {}).get("prefix_cache") or {}).get("mem_bytes", 0))
rss_pre = rss_bytes(srv_pid)
swap_pre = swap_used_bytes()
pg_size, vm_pre = vm_stat_snapshot()
swapouts_pre = vm_pre.get("Swapouts", 0)

with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
    log_before_pressure = len(fh.readlines())

stage_file = visit_dir / "hog.stage"
status_file = visit_dir / "hog.status"

def set_stage(val):
    tmp = stage_file.with_suffix(".tmp")
    tmp.write_text(str(val), encoding="utf-8")
    os.replace(tmp, stage_file)

set_stage(2)
status_file.write_text("0", encoding="utf-8")
max_hog_secs = warn_secs + crit_secs + 150.0
hog = subprocess.Popen(
    [sys.executable, "-c", HOG_CODE, str(os.getpid()), str(stage_file), str(status_file), str(max_hog_secs)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
hog_pid_file.write_text(str(hog.pid), encoding="utf-8")

swap_peak = swap_pre
kv_bytes_min = kv_bytes_pre
kv_bytes_warn = kv_bytes_pre
kv_bytes_crit = kv_bytes_pre
budget_bytes_warn = budget_bytes_pre
budget_bytes_crit = budget_bytes_pre
rss_pressure = rss_pre
peak_hog_mb = 0
reached_warn = False
reached_crit = False
warn_held_s = 0.0
crit_held_s = 0.0

try:
    t_ramp = time.monotonic() + 60.0
    while time.monotonic() < t_ramp:
        os.kill(srv_pid, 0)
        swap_peak = max(swap_peak, swap_used_bytes())
        if pressure_level() >= 2:
            reached_warn = True
            break
        if hog.poll() is not None:
            raise SystemExit(f"visit {visit} ({arm}): memory hog exited before reaching warning")
        time.sleep(0.25)
    if not reached_warn:
        raise SystemExit(f"visit {visit} ({arm}): timed out reaching warning pressure")

    warn_deadline = time.monotonic() + warn_secs + 20.0
    warn_streak = 0.0
    prev_t = time.monotonic()
    while warn_streak < warn_secs:
        if time.monotonic() >= warn_deadline:
            raise SystemExit(f"visit {visit} ({arm}): failed to sustain warning pressure for {warn_secs}s (best={warn_held_s:.1f}s)")
        if hog.poll() is not None:
            raise SystemExit(f"visit {visit} ({arm}): memory hog exited during warning hold")
        os.kill(srv_pid, 0)
        time.sleep(0.5)
        now_t = time.monotonic()
        dt = now_t - prev_t
        prev_t = now_t
        swap_peak = max(swap_peak, swap_used_bytes())
        if pressure_level() >= 2:
            warn_streak += dt
            warn_held_s = max(warn_held_s, warn_streak)
        else:
            warn_streak = 0.0

    p_warn = get_props()
    kv_bytes_warn = int(p_warn.get("memory", {}).get("kv_cache_bytes", 0))
    budget_bytes_warn = int(((p_warn.get("settings") or {}).get("prefix_cache") or {}).get("mem_bytes", 0))
    kv_bytes_min = min(kv_bytes_min, kv_bytes_warn)

    set_stage(crit_level)
    t_crit_ramp = time.monotonic() + 60.0
    while time.monotonic() < t_crit_ramp:
        os.kill(srv_pid, 0)
        swap_peak = max(swap_peak, swap_used_bytes())
        if pressure_level() >= crit_level:
            reached_crit = True
            break
        if hog.poll() is not None:
            raise SystemExit(f"visit {visit} ({arm}): memory hog exited before reaching critical")
        time.sleep(0.25)
    if not reached_crit:
        raise SystemExit(f"visit {visit} ({arm}): timed out reaching critical pressure")

    crit_deadline = time.monotonic() + crit_secs + 15.0
    crit_streak = 0.0
    prev_t = time.monotonic()
    while crit_streak < crit_secs:
        if time.monotonic() >= crit_deadline:
            raise SystemExit(f"visit {visit} ({arm}): failed to sustain critical pressure for {crit_secs}s (best={crit_held_s:.1f}s)")
        if hog.poll() is not None:
            raise SystemExit(f"visit {visit} ({arm}): memory hog exited during critical hold")
        os.kill(srv_pid, 0)
        time.sleep(0.5)
        now_t = time.monotonic()
        dt = now_t - prev_t
        prev_t = now_t
        swap_peak = max(swap_peak, swap_used_bytes())
        if pressure_level() >= crit_level:
            crit_streak += dt
            crit_held_s = max(crit_held_s, crit_streak)
        else:
            crit_streak = 0.0

    p_crit = get_props()
    kv_bytes_crit = int(p_crit.get("memory", {}).get("kv_cache_bytes", 0))
    budget_bytes_crit = int(((p_crit.get("settings") or {}).get("prefix_cache") or {}).get("mem_bytes", 0))
    kv_bytes_min = min(kv_bytes_min, kv_bytes_crit)
    rss_pressure = rss_bytes(srv_pid)
    try:
        peak_hog_mb = int(status_file.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        peak_hog_mb = 0
finally:
    set_stage(0)
    hog.terminate()
    try:
        hog.wait(timeout=10)
    except subprocess.TimeoutExpired:
        hog.kill()
        hog.wait(timeout=5)
    hog_pid_file.unlink(missing_ok=True)

_, vm_post_hog = vm_stat_snapshot()
swapouts_delta = max(0, vm_post_hog.get("Swapouts", 0) - swapouts_pre)
swap_delta_bytes = max(0, swap_peak - swap_pre)

t_settle = time.monotonic() + 60.0
while time.monotonic() < t_settle and pressure_level() != 1:
    time.sleep(0.5)
time.sleep(1.5)

with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
    lines_before_resume = fh.readlines()
resume_mark = len(lines_before_resume)

tokens_re_prefilled = 0
tokens_cached_resumed = 0
tokens_prompt_resumed = 0
resume_prompt_ms = 0.0
resume_wall_ms = 0.0

for s_idx, (msgs, reply1) in enumerate(conversations, start=1):
    msgs2 = msgs + [
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": f"Now report the tag for record {s_idx}.12 in one short sentence."},
    ]
    _, p_toks, c_toks, p_ms, w_ms = chat_turn(msgs2, max_tokens=16)
    tokens_prompt_resumed += p_toks
    tokens_cached_resumed += c_toks
    tokens_re_prefilled += max(0, p_toks - c_toks)
    resume_prompt_ms += p_ms
    resume_wall_ms += w_ms

with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
    all_lines = fh.readlines()

pressure_slice = "".join(all_lines[log_before_pressure:resume_mark])
resume_slice = "".join(all_lines[resume_mark:])

budget_steps = len(re.findall(r"\[hot-cache\] memory pressure (?:worse|eased):", pressure_slice))
worse_steps = len(re.findall(r"\[hot-cache\] memory pressure worse:", pressure_slice))
budget_revisions = len(re.findall(r"\[hot-cache\] budget revised \d+ -> \d+ MB", pressure_slice))
pressure_evictions = len(re.findall(r"\[hot-cache\] evicted LRU entry \(budget revised;", pressure_slice))

ssd_restore_ms_vals = [int(m) for m in re.findall(r"\[disk-cache\] restored \d+/\d+ tokens from SSD .* in (\d+)ms", resume_slice)]
ssd_restore_ms = sum(ssd_restore_ms_vals)
ssd_restores = len(ssd_restore_ms_vals)
ram_reuses = len(re.findall(r"\[hot-cache\] (?:full )?reuse[d]? \d+/\d+ tokens", resume_slice))

record = {
    "visit": visit,
    "arm": arm,
    "reached_warn": reached_warn,
    "reached_crit": reached_crit,
    "warn_held_s": round(warn_held_s, 2),
    "crit_held_s": round(crit_held_s, 2),
    "peak_hog_mb": peak_hog_mb,
    "budget_steps": budget_steps,
    "worse_steps": worse_steps,
    "budget_revisions": budget_revisions,
    "pressure_evictions": pressure_evictions,
    "kv_resident_pre_bytes": kv_bytes_pre,
    "kv_resident_warn_bytes": kv_bytes_warn,
    "kv_resident_crit_bytes": kv_bytes_crit,
    "kv_resident_min_bytes": kv_bytes_min,
    "mlx_active_pre_bytes": active_bytes_pre,
    "rss_pre_bytes": rss_pre,
    "rss_pressure_bytes": rss_pressure,
    "budget_pre_bytes": budget_bytes_pre,
    "budget_warn_bytes": budget_bytes_warn,
    "budget_crit_bytes": budget_bytes_crit,
    "swap_pre_bytes": swap_pre,
    "swap_peak_bytes": swap_peak,
    "swap_delta_bytes": swap_delta_bytes,
    "swapouts_delta_pages": swapouts_delta,
    "swapouts_delta_bytes": swapouts_delta * pg_size,
    "ssd_restore_ms": ssd_restore_ms,
    "ssd_restores": ssd_restores,
    "ram_reuses": ram_reuses,
    "resume_prompt_ms": round(resume_prompt_ms, 2),
    "resume_wall_ms": round(resume_wall_ms, 2),
    "seed_prompt_tokens": seed_prompt_tokens,
    "resume_prompt_tokens": tokens_prompt_resumed,
    "resume_cached_tokens": tokens_cached_resumed,
    "tokens_re_prefilled": tokens_re_prefilled,
}

with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=2)
    fh.write("\n")

mb = 1024 * 1024
print(
    f"  visit={visit} arm={arm} steps={budget_steps} (rev={budget_revisions}) "
    f"kv_mb={kv_bytes_pre / mb:.1f}->{kv_bytes_min / mb:.1f} "
    f"swap_delta_mb={swap_delta_bytes / mb:.1f} (peak={swap_peak / mb:.1f}) "
    f"restore_ms={resume_prompt_ms:.1f} (ssd_ms={ssd_restore_ms}) "
    f"re_prefilled={tokens_re_prefilled}"
)
PY

    stop_server
    rm -rf "$visit_home/.mlx-serve/kv-cache"
}

run_visit 1 base
run_visit 2 head
run_visit 3 head
run_visit 4 base

python3 - "$OUT_DIR" "$BASE_SHA" "$HEAD_SHA" "$BASE_VER" "$HEAD_VER" "$SINGLE_BIN_MODE" <<'PY'
import json, pathlib, statistics, sys

out_dir = pathlib.Path(sys.argv[1])
base_sha, head_sha, base_ver, head_ver, single_mode = sys.argv[2:7]
mb = 1024.0 * 1024.0

rows = []
for p in sorted(out_dir.glob("visit-*/metrics.json")):
    rows.append(json.loads(p.read_text(encoding="utf-8")))
if len(rows) != 4:
    raise SystemExit(f"expected 4 valid ABBA visits, found {len(rows)}")
for r in rows:
    if not (r.get("reached_warn") and r.get("reached_crit")):
        raise SystemExit(f"visit {r.get('visit')} missing required pressure levels: {r}")

tsv_cols = [
    "visit", "arm", "warn_held_s", "crit_held_s",
    "budget_steps", "budget_revisions", "pressure_evictions",
    "kv_resident_pre_bytes", "kv_resident_min_bytes", "rss_pressure_bytes",
    "swap_peak_bytes", "swap_delta_bytes", "swapouts_delta_pages",
    "ssd_restore_ms", "resume_prompt_ms", "tokens_re_prefilled", "resume_cached_tokens",
]
tsv_lines = ["\t".join(tsv_cols)]
for r in rows:
    tsv_lines.append("\t".join(str(r[c]) for c in tsv_cols))
(out_dir / "results.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

def arm_mean(arm_name, key):
    vals = [float(r[key]) for r in rows if r["arm"] == arm_name]
    return statistics.mean(vals) if vals else 0.0

md = [
    "# B3 Memory-Pressure ABBA Summary",
    "",
    f"- Mode: `{'single-binary (MLX_SERVE_PREFIX_CACHE_PRESSURE=0 vs default)' if single_mode == '1' else 'two-binaries (base vs head)'}`",
    f"- Base: `{base_ver}` (`{base_sha}`)",
    f"- Head: `{head_ver}` (`{head_sha}`)",
    "",
    "## Per-Visit (ABBA)",
    "",
    "| Visit | Arm | Hold warn/crit (s) | Budget steps | KV resident pre -> min (MB) | Server RSS under pressure (MB) | Swap delta / peak (MB) | Restore prompt_ms (SSD ms) | Tokens re-prefilled |",
    "|---:|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    md.append(
        f"| {r['visit']} | {r['arm']} | {r['warn_held_s']:.1f} / {r['crit_held_s']:.1f} | {r['budget_steps']} | "
        f"{r['kv_resident_pre_bytes'] / mb:.1f} -> {r['kv_resident_min_bytes'] / mb:.1f} | "
        f"{r['rss_pressure_bytes'] / mb:.1f} | "
        f"{r['swap_delta_bytes'] / mb:.1f} / {r['swap_peak_bytes'] / mb:.1f} | "
        f"{r['resume_prompt_ms']:.1f} ({r['ssd_restore_ms']}) | "
        f"{r['tokens_re_prefilled']} |"
    )
md += [
    "",
    "## Arm Means",
    "",
    "| Arm | Budget steps | KV resident pre (MB) | KV resident min (MB) | Server RSS under pressure (MB) | Swap delta (MB) | Swap peak (MB) | Restore prompt_ms | SSD restore (ms) | Tokens re-prefilled |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for arm_name in ("base", "head"):
    md.append(
        f"| {arm_name} | {arm_mean(arm_name, 'budget_steps'):.2f} | "
        f"{arm_mean(arm_name, 'kv_resident_pre_bytes') / mb:.1f} | "
        f"{arm_mean(arm_name, 'kv_resident_min_bytes') / mb:.1f} | "
        f"{arm_mean(arm_name, 'rss_pressure_bytes') / mb:.1f} | "
        f"{arm_mean(arm_name, 'swap_delta_bytes') / mb:.1f} | "
        f"{arm_mean(arm_name, 'swap_peak_bytes') / mb:.1f} | "
        f"{arm_mean(arm_name, 'resume_prompt_ms'):.1f} | "
        f"{arm_mean(arm_name, 'ssd_restore_ms'):.1f} | "
        f"{arm_mean(arm_name, 'tokens_re_prefilled'):.1f} |"
    )

summary_text = "\n".join(md) + "\n"
(out_dir / "summary.md").write_text(summary_text, encoding="utf-8")
print("")
print(summary_text, end="")
PY
