#!/bin/bash
# bench.sh — the performance bench. llmprobe measures; this drives mlx-serve.
#
# One `llmprobe --bench-only` run per model gives the decode/prefill/TTFT
# medians AND the context ladder. The numbers go into benchmarks.md by hand —
# there is no CSV, no chart pipeline and no engine matrix here any more.
#
#   ./tests/bench.sh                                # every model
#   ./tests/bench.sh --only qwen38-27b              # one row
#   ./tests/bench.sh --url 127.0.0.1:1234 -m <id>   # a server someone else started
#   ./tests/bench.sh --full                         # median of 3 per rung, to 64k
#
# Each cell is mlx-serve at its FASTEST: speculation is forced on where the
# checkpoint carries an MTP head (older binaries left it off on MoE). The mode
# that actually engaged is printed beside the number, from the server's own
# log — a mode that silently stops engaging shows up as a bare cell.
#
# Comparing against another engine: start it yourself (LM Studio, oMLX, MTPLX,
# llama-server, whatever), then point --url at it. Same protocol, same probe,
# one less thing in this script to keep in sync.
#
# Requirements: node (npx), curl, mlx-serve built ReleaseFast (Debug is 2-4x
# slower = a fake regression).
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ONLY=""
FULL=0
URL=""
URL_MODEL=""
TAG="$(date +%Y%m%d-%H%M%S)"
SETTLE="${SETTLE:-20}"

BINARY="${BINARY:-$ROOT/zig-out/bin/mlx-serve}"
LLMPROBE="${LLMPROBE:-npx -y llmprobe@latest}"
PORT=11250

usage() { sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)    ONLY="$2"; shift 2 ;;
        --url)     URL="$2"; shift 2 ;;
        -m|--model) URL_MODEL="$2"; shift 2 ;;
        --full)    FULL=1; shift ;;
        --tag)     TAG="$2"; shift 2 ;;
        --settle)  SETTLE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown flag: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# Reports live outside the repo: ~/claude-tmp survives reboots, /tmp does not.
OUT="$HOME/claude-tmp/bench-$TAG"
mkdir -p "$OUT"

# ── Model matrix: logical|candidates relative to a model root ──
# The first candidate found wins (tests/_lib_models.sh). A row with no checkpoint
# on this box, or one past its GPU budget, skips: a bench you can't run here
# isn't an error on the box that can.
# ANE=1 adds --ane-prefill to every boot
# (a named refusal on non-qwen3_5-dense models, so it is safe matrix-wide);
# ane-on cells are their own column, never diffed against ane-off ones.
source "$SCRIPT_DIR/_lib_models.sh"
TARGETS=(
    "gemma4-e4b-4bit|mlx-community/gemma-4-e4b-it-4bit"
    "gemma4-26b-a4b-moe-qat-4bit|mlx-community/gemma-4-26B-A4B-it-qat-4bit"
    "qwen36-35b-a3b|ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit"
    "qwen38-27b|ddalcu/Qwen3.8-27B-MLX-Serve-4bit"
    "qwen38-27b-iq|ddalcu/Qwen3.8-27B-MLX-Serve-iQ-MLX-3.8bpw"
    "qwen38-flash-next|ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
)

# Only ever called on the path that STARTED a server: --url may be pointed at
# a local mlx-serve someone else is using, and a bench must not kill it.
stop_server() {
    pkill -f "mlx-serve --serve .*--port $PORT" 2>/dev/null
    for _ in $(seq 1 30); do
        lsof -ti tcp:"$PORT" >/dev/null 2>&1 || return 0
        sleep 1
    done
}

probe() { # logical host model_id
    local depth=(--bench-only)
    [[ "$FULL" -eq 1 ]] && depth+=(--full)
    echo "── $1 ($2, $3) ──"
    # shellcheck disable=SC2086
    $LLMPROBE "$2" -m "$3" "${depth[@]}" --save "$OUT/$1.json" \
        || echo "  llmprobe failed for $1" >&2
}

# --mtp is a no-op from 26.9.7 (every loaded head drafts, MoE included), but
# older binaries left MoE heads off without it, and they are benched here too.
# A pack's own drafter/ loads on its own; a sidecar that ships separately is named here.
drafter_for() { # logical
    case "$1" in
        qwen38-27b) find_model z-lab/Qwen3.8-27B-DFlash2 ;;
    esac
}

spec_flags() { # logical model_path -> FLAGS
    FLAGS=()
    if ls "$2"/*mtp*.safetensors >/dev/null 2>&1 || [ -d "$2/mtp" ] \
       || grep -qi '"mtp' "$2/config.json" 2>/dev/null; then
        FLAGS+=(--mtp)
    fi
    local d
    if d=$(drafter_for "$1"); then FLAGS+=(--drafter "$d"); fi
    if [[ "${ANE:-0}" == "1" ]]; then FLAGS+=(--ane-prefill); fi
}

# ── Run ──
if [[ -n "$URL" ]]; then
    [[ -n "$URL_MODEL" ]] || { echo "--url needs -m <model id>" >&2; exit 1; }
    echo "=== bench: $URL ($URL_MODEL) ==="
    probe "$(echo "$URL_MODEL" | tr '/ ' '__')" "$URL" "$URL_MODEL"
else
    [[ -x "$BINARY" ]] || { echo "no $BINARY — build ReleaseFast first" >&2; exit 1; }
    echo "=== bench: mlx-serve, tag=$TAG, reports → $OUT ==="
    trap 'stop_server' EXIT
    stop_server
    for row in "${TARGETS[@]}"; do
        IFS='|' read -r logical rest <<< "$row"
        [[ -n "$ONLY" && "$logical" != *"$ONLY"* ]] && continue
        IFS='|' read -r -a cands <<< "$rest"
        path=$(find_fitting_model "${cands[@]}") || { echo "SKIP $logical (no checkpoint within $(max_model_gb) GB on this box)" >&2; continue; }
        spec_flags "$logical" "$path"
        echo; echo ">> $logical ${FLAGS[*]+${FLAGS[*]}}"
        "$BINARY" --serve --model "$path" --port "$PORT" ${FLAGS[@]+"${FLAGS[@]}"} >"$OUT/$logical.log" 2>&1 &
        pid=$!
        for _ in $(seq 1 300); do
            curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
            sleep 1
        done
        if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            probe "$logical" "localhost:$PORT" "$(basename "$path")"
        else
            echo "  mlx-serve never came up for $logical" >&2
        fi
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        stop_server
        sleep "$SETTLE"
    done
    stop_server
fi

# ── The only artifact: rows to paste into benchmarks.md ──
echo
python3 - "$OUT" <<'PY'
import json, re, sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*.json")):
    bench = (json.loads(path.read_text()) or {}).get("bench") or {}
    decode = (bench.get("decodeTokPerSec") or {}).get("median")
    prefill = (bench.get("prefillTokPerSec") or {}).get("median")
    # llmprobe leaves the top-level block null on a noisy predictable/novel pair:
    # the shortest context rung carries the same measurement.
    rungs = bench.get("contextScaling") or [{}]
    tps = ((bench.get("speculative") or {}).get("tokensPerStep")
           or (rungs[0].get("speculative") or {}).get("tokensPerStep") or 1.0)
    # WHICH speculative mode ran is only knowable from the server's own log
    # (llmprobe reports that one engaged, not which one). Name it in the cell
    # only when it actually paid: armed-but-not-accepting is not "mtp".
    log = path.with_suffix(".log")
    modes = re.findall(r"\[spec-stats\] mode=(\w+)",
                       log.read_text(errors="replace")) if log.exists() else []
    mode = f" {max(set(modes), key=modes.count)}" if modes and tps > 1.05 else ""
    if decode is None:
        print(f"| {path.stem} | · |  (no bench block)")
        continue
    pf = "n/a" if prefill is None else f"{prefill:.0f}"
    print(f"| {path.stem} | {decode:.0f}{mode} |"
          f"  (prefill {pf}, {tps:.2f} tok/step)")
PY
echo
echo "=== reports $OUT"
