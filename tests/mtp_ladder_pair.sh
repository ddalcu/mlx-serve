#!/bin/bash
# MTP ladder verdict pair: mlx-serve vs oMLX on the same oQ4e checkpoint.
#
# Protocol (BenchmarkLog "WARM protocol"): ONE boot per engine, model resident
# for all rungs; a DISCARDED 128-token warmup on a non-ladder prompt after
# boot; prompt caches disabled on both sides (--prefix-cache-entries 0 /
# --no-cache) so every rung is a cold prompt on a warm process; ours runs
# FIRST, oMLX second. Machine should be otherwise idle (no other GPU users —
# a concurrent GPU workload silently taints both lanes).
#
# Usage:
#   ./tests/mtp_ladder_pair.sh                 # both engines, full 8 rungs
#   OURS_ONLY=1 ./tests/mtp_ladder_pair.sh     # skip the oMLX lane
#   RUNGS=8192 ./tests/mtp_ladder_pair.sh      # quick single-rung probe
#   MODEL=~/.mlx-serve/models/Jundot/Qwen3.6-27B-oQ4e-mtp  # default
#
# Results: JSON per lane in $OUT_DIR (default /tmp/mtp-ladder-<date>), plus a
# side-by-side decode/prefill table on stdout. Rebuild first if you changed
# code: zig build -Doptimize=ReleaseFast (zig build test does NOT refresh the
# exe).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Same override convention as bench.sh's BINARY: point it at a released bundle's
# server (`/Applications/MLX Core.app/Contents/MacOS/mlx-serve`) to A/B this
# build against a shipped one in ONE session. Must be the in-bundle path, not a
# copy — the bundled binary resolves its dylibs via @executable_path.
BIN="${BINARY:-$ROOT/zig-out/bin/mlx-serve}"
LADDER=~/mlx-bench-assets/ladder3.py
MODEL="${MODEL:-$HOME/.mlx-serve/models/Jundot/Qwen3.6-27B-oQ4e-mtp}"
OMLX_CLI="/Applications/oMLX.app/Contents/MacOS/omlx-cli"
OMLX_MODEL_ID="${OMLX_MODEL_ID:-$(basename "$MODEL")}"
RUNGS="${RUNGS:-512,1024,2048,4096,8192,16384,32768,65536}"
PORT_OURS="${PORT_OURS:-8899}"
PORT_OMLX="${PORT_OMLX:-8890}"
OUT_DIR="${OUT_DIR:-/tmp/mtp-ladder-$(date +%Y%m%d-%H%M%S)}"
WARM_PROMPT="Write a long detailed essay about the history of optimizing compilers."
mkdir -p "$OUT_DIR"

wait_port() { # $1 url
    for _ in $(seq 1 180); do
        curl -s -m 1 "$1" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

warmup() { # $1 port, $2 body
    curl -s -m 300 "http://127.0.0.1:$1/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$2" >/dev/null
}

if pgrep -f "mlx-serve --serve" >/dev/null || pgrep -f omlx-server >/dev/null; then
    echo "FATAL: an engine is already running (mlx-serve/omlx) — stop it first."
    exit 1
fi

echo "── mlx-serve lane (first) ──"
"$BIN" --serve --model "$MODEL" --no-pld --prefix-cache-entries 0 \
    --ctx-size 140000 --port "$PORT_OURS" >"$OUT_DIR/ours-server.log" 2>&1 &
OURS_PID=$!
wait_port "http://127.0.0.1:$PORT_OURS/health" || { echo "FATAL: mlx-serve did not come up"; kill "$OURS_PID" 2>/dev/null; exit 1; }
warmup "$PORT_OURS" "{\"model\":\"mlx-serve\",\"messages\":[{\"role\":\"user\",\"content\":\"$WARM_PROMPT\"}],\"max_tokens\":128,\"temperature\":0.6,\"enable_thinking\":false}"
python3 "$LADDER" "$PORT_OURS" "$OUT_DIR/ours.json" "$RUNGS" \
    --engine mlx-serve --log ~/.mlx-serve/logs/mlx-serve-"$PORT_OURS".log --label oursjundot
kill "$OURS_PID" 2>/dev/null
wait "$OURS_PID" 2>/dev/null

if [ "${OURS_ONLY:-0}" != "1" ]; then
    echo "── oMLX lane (second) ──"
    "$OMLX_CLI" serve --model-dir "$(dirname "$MODEL")" --port "$PORT_OMLX" --no-cache \
        >"$OUT_DIR/omlx-server.log" 2>&1 &
    OMLX_PID=$!
    wait_port "http://127.0.0.1:$PORT_OMLX/v1/models" || { echo "FATAL: omlx did not come up"; kill "$OMLX_PID" 2>/dev/null; exit 1; }
    sleep 2
    warmup "$PORT_OMLX" "{\"model\":\"$OMLX_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$WARM_PROMPT\"}],\"max_tokens\":128,\"temperature\":0.6,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
    python3 "$LADDER" "$PORT_OMLX" "$OUT_DIR/omlx.json" "$RUNGS" \
        --engine omlx --model "$OMLX_MODEL_ID" --label omlx
    # omlx-cli forks omlx-server; pkill BOTH (kill by PID misses the child).
    pkill -f omlx-cli 2>/dev/null
    pkill -f omlx-server 2>/dev/null
fi

echo
python3 - "$OUT_DIR" <<'EOF'
import json, os, sys
d = sys.argv[1]
ours = json.load(open(os.path.join(d, "ours.json")))
omlx = []
p = os.path.join(d, "omlx.json")
if os.path.exists(p):
    omlx = json.load(open(p))
by = lambda rows: {r["context"]: r for r in rows}
o, m = by(ours), by(omlx)
print(f"{'ctx':>6} {'ours dec':>9} {'omlx dec':>9} {'Δdec':>7}   {'ours pp':>8} {'omlx pp':>8}")
for ctx in sorted(o):
    r, q = o[ctx], m.get(ctx)
    dd = f"{(r['decode_tok_s']/q['decode_tok_s']-1)*100:+.1f}%" if q else "-"
    print(f"{ctx:>6} {r['decode_tok_s']:>9} {q['decode_tok_s'] if q else '-':>9} {dd:>7}   "
          f"{r['client_pp_tps']:>8} {q['client_pp_tps'] if q else '-':>8}")
print(f"\nJSON: {d}/ours.json" + (f" + {d}/omlx.json" if omlx else ""))
EOF
