#!/bin/bash
# DeepSeek-V4.1 module records: controls, restart, corruption, pack stats and cancellation.
# DSV41_TEST_MODEL=<resident pack> MLX_SERVE_BINARY=<ReleaseFast binary> bash tests/test_module_prefix_cache_disk.sh
# Optional MODULE_CACHE_KV_MODEL + MODULE_CACHE_OLD_BINARY exercise KV compatibility and an older reader.
# Timing windows that cannot be observed are SKIP, never PASS. Hermetic row owners are listed by the probe.

set -euo pipefail
source "$(dirname "$0")/lib/server.sh"

MODEL="${DSV41_TEST_MODEL:-}"
[ -n "$MODEL" ] || { echo "SKIP: DSV41_TEST_MODEL not set"; exit 0; }
BINARY="${MLX_SERVE_BINARY:-${BINARY:-./zig-out/bin/mlx-serve}}"
[ -x "$BINARY" ] || { echo "FAIL: $BINARY not found. Build first."; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "FAIL: $MODEL/config.json not found"; exit 1; }
PORT="${MODULE_CACHE_TEST_PORT:-19046}"
BASE="http://127.0.0.1:$PORT"
PROBE="$(dirname "$0")/module_prefix_cache_disk_probe.py"
mkdir -p "$HOME/claude-tmp"
OUT=$(mktemp -d "$HOME/claude-tmp/module-prefix-disk.XXXXXX")
SERVER_PID=""
WATCH_PID=""
LOGFILE=""
CASE_HOME=""
EXTRA_FLAGS=()
[ -z "${DSV41_TEST_FLAGS:-}" ] || read -ra EXTRA_FLAGS <<< "$DSV41_TEST_FLAGS"

probe() { python3 "$PROBE" "$@"; }
diagnostics() {
    local log home
    for log in "$OUT"/*.server.log; do
        [ -f "$log" ] || continue
        echo "--- $log"
        cat "$log"
    done
    : > "$OUT/filesystem.txt"
    for home in "$OUT"/*-home; do
        echo "--- $home/.mlx-serve/kv-cache" >> "$OUT/filesystem.txt"
        ls -lR "$home/.mlx-serve/kv-cache" >> "$OUT/filesystem.txt" 2>&1 || true
    done
    cat "$OUT/filesystem.txt"
}
cleanup() {
    local rc=$?
    trap - EXIT
    [ "$rc" = 0 ] || diagnostics
    [ -z "$WATCH_PID" ] || kill "$WATCH_PID" 2>/dev/null || true
    if [ -n "$SERVER_PID" ]; then
        kill -CONT "$SERVER_PID" 2>/dev/null || true
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 1
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ "$rc" = 0 ]; then rm -rf "$OUT"; else echo "artifacts: $OUT"; fi
    exit "$rc"
}
trap cleanup EXIT

start_server() { # label, HOME, extra flags
    local label="$1"
    CASE_HOME="$2"
    shift 2
    mkdir -p "$CASE_HOME"
    LOGFILE="$OUT/$label.server.log"
    probe mem-wait --model "$MODEL"
    # Refuse an occupied port; never kill a server owned by another test.
    probe port-free --port "$PORT"
    HOME="$CASE_HOME" "$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
        --ctx-size 16384 --no-pld --no-mtp --no-drafter --no-vision --kv-quant off --metrics \
        --prefix-cache-entries 4 --prefix-cache-mem 8GB --prefix-cache-disk 32GB \
        --shutdown-persist-secs 45 --log-level debug \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} "$@" > "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    srv_wait_health "$BASE" 1200 1 "$SERVER_PID"
}
stop_server() {
    # Shutdown persistence is bounded; Writer teardown can still join an in-flight write.
    probe stop --pid "$SERVER_PID" --timeout 120
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
request() { probe request --base "$BASE" "$@"; }
copy_seed() { probe copy-cache --source "$OUT/seed-home" --dest "$1"; }

# ── Tests ──

echo "entries zero and module RAM disabled retain no history"
probe fixture --model "$MODEL" --dest "$OUT/pack"
MODEL="$OUT/pack"
start_server r02-zero "$OUT/zero-home" --prefix-cache-entries 0
request --output "$OUT/cold1.json"
request --seed "$OUT/cold1.json" --output "$OUT/cold2.json"
probe result "$OUT/cold1.json" --cached 0
probe result "$OUT/cold2.json" --cached 0
stop_server
probe disabled --root "$OUT/zero-home" --log "$LOGFILE" --reason
start_server r02-no-ram "$OUT/no-ram-home" --no-prefix-cache-ram
request --output "$OUT/no-ram1.json"
request --seed "$OUT/cold1.json" --output "$OUT/no-ram2.json"
probe result "$OUT/no-ram2.json" --cached 0 --cold "$OUT/cold2.json"
stop_server
probe disabled --root "$OUT/no-ram-home" --log "$LOGFILE" --reason
echo "PASS: entries zero and module RAM disabled retain no history"

echo "disk disabled preserves exact RAM restore"
start_server r01 "$OUT/off-home" --prefix-cache-disk off
request --output "$OUT/off1.json"
request --seed "$OUT/cold1.json" --output "$OUT/off2.json"
probe result "$OUT/off2.json" --boundary "$OUT/cold1.json" --cold "$OUT/cold2.json" --log "$LOGFILE" --source ram
stop_server
probe disabled --root "$OUT/off-home" --log "$LOGFILE"
echo "PASS: disk disabled preserves exact RAM restore"

echo "clean shutdown publishes v10 and restart restores exact position"
start_server r18-seed "$OUT/seed-home"
request --output "$OUT/seed1.json"
probe result "$OUT/seed1.json" --cached 0 --cold "$OUT/cold1.json"
stop_server
probe summary --log "$LOGFILE" --deadline 45
probe layout --root "$OUT/seed-home" --boundary "$OUT/seed1.json" --snapshot "$OUT/seed-records.json"
copy_seed "$OUT/restart-home"
start_server r25-restart "$OUT/restart-home"
request --seed "$OUT/cold1.json" --output "$OUT/restart2.json"
probe result "$OUT/restart2.json" --boundary "$OUT/seed1.json" --cold "$OUT/cold2.json" --log "$LOGFILE" --source disk
echo "PASS: clean shutdown publishes v10 and restart restores exact position"

echo "newer checkpoint preserves the older immutable record"
stop_server
probe preserved --root "$OUT/restart-home" --snapshot "$OUT/seed-records.json"
probe layout --root "$OUT/restart-home" --boundary "$OUT/restart2.json"
echo "PASS: newer checkpoint preserves the older immutable record"

echo "truncated tensor part rejects the whole record"
copy_seed "$OUT/truncated-home"
probe truncate --root "$OUT/truncated-home" --boundary "$OUT/seed1.json"
start_server r17 "$OUT/truncated-home"
probe dropped --log "$LOGFILE" --snapshot "$OUT/seed-records.json"
request --seed "$OUT/cold1.json" --output "$OUT/truncated2.json"
probe result "$OUT/truncated2.json" --cached 0 --cold "$OUT/cold2.json"
stop_server
echo "PASS: truncated tensor part rejects the whole record"

echo "changed pack stat selects a new module root"
copy_seed "$OUT/stamp-home"
if probe touch-pack --pack "$MODEL"; then
    start_server r19 "$OUT/stamp-home"
    request --seed "$OUT/cold1.json" --output "$OUT/stamp2.json"
    probe result "$OUT/stamp2.json" --cached 0 --cold "$OUT/cold2.json"
    stop_server
    probe changed-root --root "$OUT/stamp-home" --snapshot "$OUT/seed-records.json"
    probe restore-stat --pack "$MODEL"
    echo "PASS: changed pack stat selects a new module root"
else
    rc=$?
    [ "$rc" = 77 ] || exit "$rc"
fi

echo "SIGKILL before metadata leaves no accepted record"
start_server r16-seed "$OUT/crash-home"
probe crash-window --pid "$SERVER_PID" --root "$CASE_HOME" --output "$OUT/crash-window.json" &
WATCH_PID=$!
request --output "$OUT/crash1.json" > "$OUT/crash-client.log" 2>&1 || true
if wait "$WATCH_PID"; then
    WATCH_PID=""
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    start_server r16-restart "$OUT/crash-home"
    probe crash-scan --root "$CASE_HOME" --log "$LOGFILE" --window "$OUT/crash-window.json"
    request --output "$OUT/crash-restart1.json"
    probe result "$OUT/crash-restart1.json" --cached 0 --cold "$OUT/cold1.json"
    stop_server
    echo "PASS: SIGKILL before metadata leaves no accepted record"
else
    rc=$?
    WATCH_PID=""
    [ "$rc" = 77 ] || exit "$rc"
    stop_server
fi

echo "disabled shutdown persistence; deadline summary is honest"
for deadline in 0 1; do
    start_server "r37-r38-$deadline" "$OUT/deadline-$deadline-home" --shutdown-persist-secs "$deadline"
    request --output "$OUT/deadline-$deadline.json"
    stop_server
    probe summary --log "$LOGFILE" --deadline "$deadline" --root "$CASE_HOME"
done
echo "PASS: shutdown log controls (expiry during a slice remains unit-owned)"

echo "cancelling a continuation preserves the consumed dirty export"
start_server r47-seed "$OUT/cancel-home"
if probe cancel --base "$BASE" --root "$CASE_HOME" --log "$LOGFILE" --output "$OUT/cancel1.json"; then
    probe layout --root "$CASE_HOME" --boundary "$OUT/cancel1.json" --wait 120
    stop_server
    probe layout --root "$CASE_HOME" --boundary "$OUT/cancel1.json" --only
    start_server r47-restart "$OUT/cancel-home"
    request --max-tokens 1 --output "$OUT/cancel-restart1.json"
    probe result "$OUT/cancel-restart1.json" --boundary "$OUT/cancel1.json" --cold "$OUT/cancel1.json" --log "$LOGFILE" --source disk
    stop_server
    echo "PASS: cancelling a continuation preserves the consumed dirty export"
else
    rc=$?
    [ "$rc" = 77 ] || exit "$rc"
    stop_server
fi

if [ -n "${MODULE_CACHE_KV_MODEL:-}" ]; then
    echo "new reader restores persisted KV and rejects module records in its KV root"
    MODEL="$MODULE_CACHE_KV_MODEL"
    start_server r42-seed "$OUT/kv-home"
    request --output "$OUT/kv1.json"
    stop_server
    probe inject-module --root "$OUT/kv-home" --source "$OUT/seed-home" --output "$OUT/injected.json"
    start_server r42-new-reader "$OUT/kv-home"
    request --output "$OUT/kv-new.json"
    probe kv-result "$OUT/kv-new.json" --cold "$OUT/kv1.json" --log "$LOGFILE" --injected "$OUT/injected.json"
    stop_server
    echo "PASS: new reader restores KV and rejects a v10 module in its KV root"
    if [ -n "${MODULE_CACHE_OLD_BINARY:-}" ]; then
        echo "older reader rejects v10 without losing its KV restore"
        probe inject-module --root "$OUT/kv-home" --source "$OUT/seed-home" --output "$OUT/injected-old.json"
        BINARY="$MODULE_CACHE_OLD_BINARY"
        start_server r41-old-reader "$OUT/kv-home"
        request --output "$OUT/kv-old.json"
        probe kv-result "$OUT/kv-old.json" --cold "$OUT/kv1.json" --log "$LOGFILE" --injected "$OUT/injected-old.json"
        stop_server
        echo "PASS: older reader rejects v10 and still restores KV"
    else
        echo 'SKIP: set MODULE_CACHE_OLD_BINARY to a v2-v9-only reader'
    fi
else
    echo 'SKIP: set MODULE_CACHE_KV_MODEL to a small KV-cache pack'
fi
probe unit-owners
echo "PASS test_module_prefix_cache_disk (inspect SKIP lines for unobserved branches)"
