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
    # `|| true`: under `set -e` a failing kill (already reaped) would end an EXIT trap with status 1.
    kill "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
}

# The checkout this helper lives in. Read at source time: inside a zsh function $0 is the function's name.
_SRV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd -P)"

# srv_preclean <port>: free <port> for this script's server. Stops the mlx-serve LISTENING on it when that server was
# started from this checkout (a leftover run, or this script's own) and waits for it to exit, so its memory is back.
# Any other holder ends the script: it would answer /health in our server's place. Never matches by name or flags,
# which also takes a server on another port.
srv_preclean() {
    local pid cmd cwd _
    for pid in $(lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null); do
        cmd="$(ps -p "$pid" -o comm= 2>/dev/null)"
        cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')"
        case "${cmd##*/}:$cwd/" in
            mlx-serve:"$_SRV_ROOT"/*) kill "$pid" 2>/dev/null || true ;;
            *) echo "FAIL: port $1 is held by pid $pid ($cmd, started in ${cwd:-?}), not a server from this checkout:" \
                    "stop it or pass another port" >&2; exit 1 ;;
        esac
        for _ in $(seq 1 300); do
            # Our own child stays a zombie (kill -0 succeeds) until the script reaps it.
            case "$(ps -p "$pid" -o stat= 2>/dev/null)" in "" | Z*) continue 2 ;; esac
            sleep 0.1
        done
        kill -9 "$pid" 2>/dev/null || true
    done
    return 0
}
