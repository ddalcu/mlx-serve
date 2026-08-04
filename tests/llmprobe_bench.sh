#!/bin/bash
# Four-engine llmprobe bench: mlx-serve vs MTPLX vs oMLX vs LM Studio, one at a
# time on the same checkpoint. Per engine: boot, `npx llmprobe@latest --quick
# --bench`, save JSON, kill, 20 s settle. Ends with llmprobe's --compare page.
#
# Usage:
#   ./tests/llmprobe_bench.sh                      # all four engines
#   ENGINES="mlx-serve mtplx" ./tests/llmprobe_bench.sh
#   MODEL=~/.mlx-serve/models/Org/Name ./tests/llmprobe_bench.sh
#
# Engines with a missing binary/app are skipped with a note, not an error.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-$HOME/.mlx-serve/models/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed}"
MODEL_ID="$(basename "$MODEL")"
ENGINES="${ENGINES:-mlx-serve mtplx omlx lmstudio}"
OUT="${OUT:-$HOME/claude-tmp/llmprobe-bench-$(date +%Y%m%d-%H%M%S)}"
SETTLE="${SETTLE:-20}"

MLXSERVE_BIN="${MLXSERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
# Prefer the repo checkout's synced venv (tracks the tag you checked out);
# fall back to the installed app runtime.
MTPLX_BIN="${MTPLX_BIN:-$HOME/projects/agents/MTPLX/.venv/bin/mtplx}"
[ -x "$MTPLX_BIN" ] || MTPLX_BIN="$HOME/.mtplx/bin/mtplx"
OMLX_BIN="${OMLX_BIN:-/Applications/oMLX.app/Contents/MacOS/omlx-cli}"
# LM Studio serves its own catalog: this is an LMS model key, not a path.
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-qwen3.6-27b}"

PORT_MLXSERVE=11250 PORT_MTPLX=11252 PORT_OMLX=11251 PORT_LMS=1234

if pgrep -f "mlx-serve --serve" >/dev/null || pgrep -f "mtplx serve" >/dev/null \
    || pgrep -f omlx-server >/dev/null; then
    echo "FATAL: an engine is already running — stop it first." >&2
    exit 1
fi
mkdir -p "$OUT"

wait_url() { # $1 url  $2 tries
    for _ in $(seq 1 "$2"); do
        curl -sf -m 2 "$1" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

# JIT-load warmup doubling as a load-completed probe (LM Studio / oMLX load on
# first request; the others load at boot and pass through instantly).
warmup() { # $1 port  $2 model id
    curl -sf -m 600 "http://127.0.0.1:$1/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false}" \
        >/dev/null 2>&1
}

probe() { # $1 engine  $2 port  $3 model id
    echo "── llmprobe: $1 (:$2, model $3) ──"
    npx -y llmprobe@latest "localhost:$2" -m "$3" --quick --bench \
        --save "$OUT/$1.json" || echo "  llmprobe failed for $1" >&2
}

for engine in $ENGINES; do
    PID=""
    case "$engine" in
        mlx-serve)
            [ -x "$MLXSERVE_BIN" ] || { echo "skip mlx-serve: no $MLXSERVE_BIN"; continue; }
            "$MLXSERVE_BIN" --serve --model "$MODEL" --port "$PORT_MLXSERVE" \
                >"$OUT/mlx-serve-server.log" 2>&1 &
            PID=$!
            wait_url "http://127.0.0.1:$PORT_MLXSERVE/health" 180 || { echo "mlx-serve never came up"; kill "$PID" 2>/dev/null; continue; }
            probe mlx-serve "$PORT_MLXSERVE" "$MODEL_ID"
            kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
            ;;
        mtplx)
            [ -x "$MTPLX_BIN" ] || { echo "skip mtplx: no mtplx binary"; continue; }
            "$MTPLX_BIN" serve --model "$MODEL" --port "$PORT_MTPLX" \
                >"$OUT/mtplx-server.log" 2>&1 &
            PID=$!
            wait_url "http://127.0.0.1:$PORT_MTPLX/health" 240 || { echo "mtplx never came up"; kill "$PID" 2>/dev/null; continue; }
            MTPLX_ID="$(curl -sf "http://127.0.0.1:$PORT_MTPLX/health" | python3 -c 'import json,sys; print(json.load(sys.stdin)["model"])' 2>/dev/null || echo "$MODEL_ID")"
            probe mtplx "$PORT_MTPLX" "$MTPLX_ID"
            pkill -f "mtplx serve" 2>/dev/null
            ;;
        omlx)
            [ -x "$OMLX_BIN" ] || { echo "skip omlx: no $OMLX_BIN"; continue; }
            "$OMLX_BIN" serve --model-dir "$(dirname "$MODEL")" --port "$PORT_OMLX" \
                >"$OUT/omlx-server.log" 2>&1 &
            PID=$!
            wait_url "http://127.0.0.1:$PORT_OMLX/v1/models" 240 || { echo "omlx never came up"; kill "$PID" 2>/dev/null; continue; }
            warmup "$PORT_OMLX" "$MODEL_ID" || { echo "omlx warmup failed"; }
            probe omlx "$PORT_OMLX" "$MODEL_ID"
            # omlx-cli forks omlx-server; kill both.
            pkill -f omlx-cli 2>/dev/null; pkill -f omlx-server 2>/dev/null
            ;;
        lmstudio)
            command -v lms >/dev/null || { echo "skip lmstudio: no lms CLI"; continue; }
            lms server start --port "$PORT_LMS" >/dev/null 2>&1
            wait_url "http://127.0.0.1:$PORT_LMS/v1/models" 60 || { echo "lmstudio never came up"; continue; }
            lms unload --all >/dev/null 2>&1
            # `lms load` silently hangs on some releases — HTTP JIT-load is the
            # supported path (tests/bench.sh, same recipe).
            warmup "$PORT_LMS" "$LMSTUDIO_MODEL" || { echo "lmstudio JIT-load failed for $LMSTUDIO_MODEL"; lms server stop >/dev/null 2>&1; continue; }
            probe lmstudio "$PORT_LMS" "$LMSTUDIO_MODEL"
            lms unload --all >/dev/null 2>&1
            lms server stop >/dev/null 2>&1
            ;;
        *)
            echo "skip unknown engine: $engine" ;;
    esac
    echo "settling ${SETTLE}s..."
    sleep "$SETTLE"
done

# One page, all engines overlaid — llmprobe's own comparison report.
saved=$(ls "$OUT"/*.json 2>/dev/null | grep -v server)
if [ "$(echo "$saved" | grep -c .)" -ge 2 ]; then
    # shellcheck disable=SC2086
    npx -y llmprobe@latest --compare $saved --html "$OUT/compare.html" \
        && echo "comparison page: $OUT/compare.html"
fi
echo "results in $OUT"
