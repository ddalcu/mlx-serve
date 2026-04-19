#!/bin/bash
# Three-way backend comparison driver.
#
# For each backend in {mlx-serve, mlx-lm, llama.cpp}:
#   1. start server on :8080 with the Qwen3.6-35B-A3B weights
#   2. wait for /v1/models
#   3. run pi_nonstream_smoke.sh with enable_thinking=false and =true
#   4. run pi_integration_run.sh against the already-running server (quick matrix, qwen case)
#   5. collect metrics into compare_backends.tsv
#   6. stop the server
#
# Writes:
#   tests/pi-results/compare_<backend>.server.log
#   tests/pi-results/compare_<backend>.smoke-nothink.log
#   tests/pi-results/compare_<backend>.smoke-think.log
#   tests/pi-results/compare_<backend>.agent.log
#   tests/compare_backends.tsv
#   tests/compare_backends_results.md  (appended; written from scratch on first run)
#
# Usage: tests/compare_backends_run.sh [backend ...]
#   Default: mlx-serve mlx-lm llama.cpp
#   Pass a subset to run only those, e.g.: tests/compare_backends_run.sh mlx-serve

set -u
set -o pipefail

REPO="/Users/david/projects/agents/mlx-serve"
RESULTS="$REPO/tests/pi-results"
TSV="$REPO/tests/compare_backends.tsv"
MD="$REPO/tests/compare_backends_results.md"

MLX_BIN="$REPO/app/MLX Core.app/Contents/MacOS/mlx-serve"
MLX_MODEL_DIR="$HOME/.mlx-serve/models/Qwen3.6-35B-A3B-6bit"
LLAMA_BIN="$HOME/.mlx-serve/backends/llama.cpp/bin/llama-server"
GGUF_PATH="$HOME/.mlx-serve/backends/gguf/Qwen_Qwen3.6-35B-A3B-Q6_K.gguf"

PORT=8080
BASE="http://127.0.0.1:$PORT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

BACKENDS=("$@")
if [ "${#BACKENDS[@]}" -eq 0 ]; then
    BACKENDS=("mlx-serve" "mlx-lm" "llama.cpp")
fi

mkdir -p "$RESULTS"

say() { printf "${YELLOW}[compare]${NC} %s\n" "$*"; }
ok()  { printf "${GREEN}[ok]${NC}      %s\n" "$*"; }
err() { printf "${RED}[fail]${NC}    %s\n" "$*" >&2; }

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
for b in "${BACKENDS[@]}"; do
    case "$b" in
        mlx-serve)
            [ -x "$MLX_BIN" ] || { err "mlx-serve binary missing: $MLX_BIN"; exit 1; }
            [ -d "$MLX_MODEL_DIR" ] || { err "MLX weights missing: $MLX_MODEL_DIR"; exit 1; }
            ;;
        mlx-lm)
            command -v mlx_lm.server >/dev/null || { err "mlx_lm.server missing — pip install mlx-lm"; exit 1; }
            [ -d "$MLX_MODEL_DIR" ] || { err "MLX weights missing: $MLX_MODEL_DIR"; exit 1; }
            ;;
        llama.cpp)
            [ -x "$LLAMA_BIN" ] || { err "llama-server missing — run tests/compare_backends_setup.sh"; exit 1; }
            [ -f "$GGUF_PATH" ] || { err "GGUF missing — run tests/compare_backends_setup.sh"; exit 1; }
            ;;
        *)
            err "unknown backend: $b"; exit 1 ;;
    esac
done

# Port must be free.
if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    err "port $PORT already in use. Free it before running."
    lsof -iTCP:$PORT -sTCP:LISTEN
    exit 1
fi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Kill everything that might still be holding :$PORT.
kill_any_server() {
    pkill -f "MLX Core.app/Contents/MacOS/mlx-serve" 2>/dev/null || true
    pkill -f "mlx_lm.server" 2>/dev/null || true
    pkill -f "llama-server" 2>/dev/null || true
    for _ in $(seq 1 20); do
        if ! lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# Wait up to 5 min for /health (mlx-serve + llama.cpp) or /v1/models (mlx-lm
# which does not ship /health on older versions). We try both.
wait_ready() {
    local deadline=$(( $(date +%s) + 300 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -sf "$BASE/health" >/dev/null 2>&1; then return 0; fi
        if curl -sf "$BASE/v1/models" >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    return 1
}

# Sample peak RSS of a single pid every 5s. Writes max GB to stdout when the
# pid dies. All three backends run as a single process (no child workers),
# so single-pid tracking is accurate enough. Intended to run in the background.
rss_monitor() {
    local pid="$1" maxkb=0
    while kill -0 "$pid" 2>/dev/null; do
        local cur
        cur=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -n "$cur" ] && [ "$cur" -gt "$maxkb" ] && maxkb=$cur
        sleep 5
    done
    python3 -c "print(round($maxkb/1024/1024, 2))"
}

# Fetch the model id advertised by the server (used as pi's model id).
fetch_model_id() {
    curl -sf "$BASE/v1/models" 2>/dev/null \
        | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["data"][0]["id"])' \
        2>/dev/null || echo ""
}

# -----------------------------------------------------------------------------
# Backend starters. Each echoes the PID on stdout and writes stderr/stdout to
# the given log file.
# -----------------------------------------------------------------------------

start_mlx_serve() {
    local logfile="$1"
    "$MLX_BIN" --model "$MLX_MODEL_DIR" --serve --port "$PORT" \
        --log-level info --ctx-size 32768 \
        > "$logfile" 2>&1 &
    echo $!
}

start_mlx_lm() {
    local logfile="$1"
    mlx_lm.server --model "$MLX_MODEL_DIR" \
        --host 127.0.0.1 --port "$PORT" \
        --max-tokens 8192 \
        > "$logfile" 2>&1 &
    echo $!
}

start_llama_cpp() {
    local logfile="$1"
    "$LLAMA_BIN" -m "$GGUF_PATH" --host 127.0.0.1 --port "$PORT" \
        -c 32768 --jinja --no-webui -ngl 999 -n 8192 \
        > "$logfile" 2>&1 &
    echo $!
}

# -----------------------------------------------------------------------------
# One backend end-to-end.
# -----------------------------------------------------------------------------
run_backend() {
    local backend="$1"
    local slug="${backend//./_}"           # safe filename slug
    local server_log="$RESULTS/compare_${slug}.server.log"
    local smoke_nothink_log="$RESULTS/compare_${slug}.smoke-nothink.log"
    local smoke_think_log="$RESULTS/compare_${slug}.smoke-think.log"
    local agent_log="$RESULTS/compare_${slug}.agent.log"
    : > "$server_log" "$smoke_nothink_log" "$smoke_think_log" "$agent_log"

    say "===== $backend ====="
    kill_any_server

    local load_start load_end pid
    load_start=$(date +%s)
    case "$backend" in
        mlx-serve)  pid=$(start_mlx_serve  "$server_log") ;;
        mlx-lm)     pid=$(start_mlx_lm     "$server_log") ;;
        llama.cpp)  pid=$(start_llama_cpp  "$server_log") ;;
    esac
    say "server PID=$pid (log: $server_log)"

    # Start RSS monitor in the background; its stdout is the final peak GB.
    local rss_file="$RESULTS/compare_${slug}.rss.txt"
    ( rss_monitor "$pid" > "$rss_file" ) &
    local mon_pid=$!

    if ! wait_ready; then
        err "$backend failed to become ready"
        kill "$pid" 2>/dev/null || true
        wait "$mon_pid" 2>/dev/null || true
        printf "%s\t%s\tserver-timeout\t\t\t\t\t\t\t\n" \
            "$(date +%Y-%m-%dT%H:%M:%S)" "$backend" >> "$TSV"
        return 1
    fi
    load_end=$(date +%s)
    local load_s=$(( load_end - load_start ))
    ok "$backend ready in ${load_s}s"

    local model_id
    model_id=$(fetch_model_id)
    say "advertised model id: ${model_id:-<empty>}"

    # ----- smoke: enable_thinking=false ----------------------------------
    say "smoke (think=off)"
    bash "$REPO/tests/pi_nonstream_smoke.sh" "$PORT" false > "$smoke_nothink_log" 2>&1 || true
    # ----- smoke: enable_thinking=true -----------------------------------
    say "smoke (think=on)"
    bash "$REPO/tests/pi_nonstream_smoke.sh" "$PORT" True > "$smoke_think_log" 2>&1 || true

    parse_smoke() {
        # Emits: "<tool_calls>|<ok_json_count>|<tag_leak>|<prompt>|<completion>|<elapsed>"
        local f="$1"
        python3 - "$f" <<'PY'
import re, sys
p = sys.argv[1]
try:
    t = open(p).read()
except Exception:
    print("?|?|?|?|?|?"); sys.exit(0)
def m(pat, default="?"):
    r = re.search(pat, t)
    return r.group(1) if r else default
tcs = m(r"tool_calls=(\d+)")
ok_json = len(re.findall(r"ok_json\b", t))
leak = m(r"tag_leak_in_content=(True|False)")
prompt = m(r"prompt=(\d+)")
compl = m(r"completion=(\d+)")
elapsed = m(r"elapsed=([\d.]+)")
print(f"{tcs}|{ok_json}|{leak}|{prompt}|{compl}|{elapsed}")
PY
    }
    local smoke_off smoke_on
    smoke_off=$(parse_smoke "$smoke_nothink_log")
    smoke_on=$(parse_smoke "$smoke_think_log")
    ok "smoke off: $smoke_off"
    ok "smoke on : $smoke_on"

    # ----- agent: reuse pi_integration_run.sh quick matrix ---------------
    # pi_integration_run.sh runs every case in its matrix; "quick" is e4b only.
    # We want only the qwen case but with the already-running server, so we
    # craft a one-case invocation by setting MATRIX=all and filtering — but
    # the simpler route is to call run_one_case directly via bash -c sourcing.
    # Instead we copy the minimal shell-invocation here to avoid plumbing a
    # new matrix name into the shared script.
    say "agent harness (pi_integration_run.sh, qwen case only)"
    # We invoke the full script with SKIP_SERVER_START=1 and MATRIX=qwen-only;
    # to get the qwen case to run we use a small here-doc wrapper that sources
    # the existing script's helpers and calls run_one_case directly.
    local agent_wrapper="$RESULTS/compare_${slug}.wrapper.sh"
    cat > "$agent_wrapper" <<WRAP
#!/bin/bash
set -u
export SKIP_SERVER_START=1
export PORT=$PORT
# All three backends ignore the model field when only one model is loaded,
# so we send the same synthetic id everywhere to keep pi's config consistent.
export SERVED_MODEL="qwen3_5_moe"
# Reduce the matrix to just the qwen-no-think case by stripping the other
# cases from a copy of the driver. BRE syntax only — portable on BSD/GNU sed.
TMPDRV="\$(mktemp)"
sed -e '/^CASES+=("e4b-stream|/d' \
    -e '/^    CASES+=("a4b-stream|/d' \
    -e '/^    CASES+=("qwen-think|/d' \
    "$REPO/tests/pi_integration_run.sh" > "\$TMPDRV"
chmod +x "\$TMPDRV"
bash "\$TMPDRV" all
rm -f "\$TMPDRV"
WRAP
    chmod +x "$agent_wrapper"
    ( bash "$agent_wrapper" ) > "$agent_log" 2>&1 || true

    local agent_score agent_notes
    agent_score=$(grep -oE 'SCORE: [0-9]+/5' "$agent_log" | tail -1 | awk '{print $2}')
    agent_notes=$(grep -oE 'SCORE: [0-9]+/5 .*' "$agent_log" | tail -1 | sed -E 's|SCORE: [0-9]+/5 ||' | sed -E 's|\[total=[0-9]+s\]||' | tr -s ' ')
    agent_score="${agent_score:-NA/5}"

    # ----- teardown ------------------------------------------------------
    local total_s=$(( $(date +%s) - load_start ))
    kill "$pid" 2>/dev/null || true
    # Give it 10 s to die gracefully; then SIGKILL.
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
    kill_any_server

    local peak_gb
    peak_gb=$(cat "$rss_file" 2>/dev/null || echo "?")

    # ----- write TSV row -------------------------------------------------
    local ts
    ts=$(date +%Y-%m-%dT%H:%M:%S)
    # columns: timestamp backend load_s peak_gb
    #          smoke_off(tcs|ok_json|leak|prompt|completion|elapsed)
    #          smoke_on (same)
    #          agent_score total_s notes
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$ts" "$backend" "$load_s" "$peak_gb" \
        "$smoke_off" "$smoke_on" \
        "$agent_score" "$total_s" "$agent_notes" >> "$TSV"

    ok "$backend done (load=${load_s}s, peak=${peak_gb}GB, agent=$agent_score, total=${total_s}s)"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if [ ! -f "$TSV" ]; then
    printf "timestamp\tbackend\tload_s\tpeak_rss_gb\tsmoke_nothink\tsmoke_think\tagent_score\ttotal_s\tnotes\n" > "$TSV"
fi

for b in "${BACKENDS[@]}"; do
    run_backend "$b" || err "$b run failed, continuing"
done

kill_any_server

# -----------------------------------------------------------------------------
# Render results markdown (appends a table run; human fills in analysis)
# -----------------------------------------------------------------------------
python3 - <<'PY' "$TSV" "$MD"
import sys, os, datetime
tsv, md = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
if not rows or len(rows) < 2:
    sys.exit(0)
header = rows[0]
# Keep only rows from this run: last N rows where N = number of backends just
# written. We approximate that as "all rows with the most recent timestamp
# minute" — good enough for the driver's append-only semantics.
recent_ts = sorted({r[0] for r in rows[1:]})[-3:]
recent = [r for r in rows[1:] if r[0] in recent_ts]

def fmt_smoke(s):
    # tcs|ok_json|leak|prompt|completion|elapsed
    parts = s.split("|")
    if len(parts) != 6: return s
    tcs, ok_json, leak, prompt, compl, el = parts
    return f"tcs={tcs} okJson={ok_json} leak={leak} prompt={prompt} compl={compl} {el}s"

lines = []
lines.append("")
lines.append(f"## Run {datetime.datetime.now().isoformat(timespec='seconds')}")
lines.append("")
lines.append("| Backend | Load (s) | Peak RSS (GB) | Smoke think=off | Smoke think=on | Agent score | Total (s) |")
lines.append("|---|---|---|---|---|---|---|")
for r in recent:
    # r matches header order
    d = dict(zip(header, r))
    lines.append("| {backend} | {load_s} | {peak} | {so} | {son} | {as_} | {tot} |".format(
        backend=d.get("backend",""),
        load_s=d.get("load_s",""),
        peak=d.get("peak_rss_gb",""),
        so=fmt_smoke(d.get("smoke_nothink","")),
        son=fmt_smoke(d.get("smoke_think","")),
        as_=d.get("agent_score",""),
        tot=d.get("total_s",""),
    ))
lines.append("")
lines.append("Notes column (per backend):")
lines.append("")
for r in recent:
    d = dict(zip(header, r))
    lines.append(f"- **{d.get('backend','')}**: {d.get('notes','').strip()}")
lines.append("")

# Prepend header + analysis scaffold only if the file is missing or empty.
existing = ""
if os.path.exists(md):
    existing = open(md).read()
if not existing.strip():
    preamble = [
        "# Three-way serving-backend comparison",
        "",
        "Target model: **Qwen3.6-35B-A3B** (35B MoE, 3B active).",
        "",
        "- **mlx-serve**: our Zig server, MLX 6-bit affine (group 64), ~27 GB.",
        "- **mlx-lm**: Apple's reference Python server, same MLX 6-bit weights.",
        "- **llama.cpp**: upstream C++ server, GGUF Q6_K (~30 GB).",
        "",
        "Raw metrics live in `tests/compare_backends.tsv`. Per-backend logs live in",
        "`tests/pi-results/compare_<backend>.*`. This file is appended by",
        "`tests/compare_backends_run.sh` and then hand-edited for analysis.",
        "",
        "## Analysis (hand-written — edit after each run)",
        "",
        "### 1. mlx-serve vs mlx-lm on identical weights",
        "_TODO after run: any gap is an implementation gap in our Zig code_",
        "",
        "### 2. Tool-call reliability with enable_thinking=false",
        "_TODO after run: parse-success across three backends; garbage shapes_",
        "",
        "### 3. Thinking-tag streaming",
        "_TODO after run: reasoning_content vs <think> in-band vs none_",
        "",
        "### 4. Agent-loop quality on the same pi harness",
        "_TODO after run: scores should bracket ±1; gaps >2 points indicate real issues_",
        "",
        "## Runs",
    ]
    with open(md, "w") as f:
        f.write("\n".join(preamble) + "\n")

with open(md, "a") as f:
    f.write("\n".join(lines) + "\n")
PY

printf "${GREEN}Done.${NC} TSV: $TSV\nResults doc: $MD\n"
cat "$TSV"
