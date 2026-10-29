#!/bin/bash
# Boot-and-poll helpers for the live test scripts. Each script still starts its own server (its flags,
# its log, its scratch HOME) and kills only that server; these replace the copied wait loops.
#
#   source "$(dirname "$0")/lib/server.sh"
#   "$BIN" --serve --port "$PORT" ... >"$LOG" 2>&1 & SRV=$!
#   srv_wait_health "http://127.0.0.1:$PORT" 120 0.5 "$SRV" || { echo "FAIL: no server"; tail -5 "$LOG"; exit 1; }
#
# Readiness is the STATUS: /health answers 503 "loading" while a model loads, so a bare socket check lies.

# srv_wait_health <base url> <tries> <interval s> [pid]
# 0 once <base>/health answers 200; 1 when [pid] has exited or the tries run out.
srv_wait_health() {
    local base="$1" tries="$2" interval="$3" pid="${4:-}" _
    for _ in $(seq 1 "$tries"); do
        curl -sf -m 5 "$base/health" >/dev/null 2>&1 && return 0
        [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && return 1
        sleep "$interval"
    done
    return 1
}

# srv_wait_log <log file> <grep pattern> <tries> <interval s> [pid]
# 0 once the log carries the pattern (a load's "Model ready" line); 1 when [pid] has exited or the tries run out.
srv_wait_log() {
    local log="$1" pattern="$2" tries="$3" interval="$4" pid="${5:-}" _
    for _ in $(seq 1 "$tries"); do
        grep -q "$pattern" "$log" 2>/dev/null && return 0
        [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && return 1
        sleep "$interval"
    done
    return 1
}

# srv_stop <pid>: TERM the server this script started and reap it. Never kills by name or binary path.
srv_stop() {
    [ -n "${1:-}" ] || return 0
    kill "$1" 2>/dev/null
    wait "$1" 2>/dev/null
    return 0
}
