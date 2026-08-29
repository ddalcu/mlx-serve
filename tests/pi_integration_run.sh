#!/bin/bash
# pi ↔ mlx-serve integration test driver.
#
# Tests every model × streaming × thinking combo by pointing the `pi`
# (https://github.com/badlogic/pi-mono) coding agent at a running mlx-serve
# instance and having it build + test a tiny express todo app.
#
# Requires: pi (npm -g @mariozechner/pi-coding-agent), node, python3.
#
# Usage: tests/pi_integration_run.sh [matrix]
#   matrix=all (default): every model × configured mode, express-todo scenario
#   matrix=quick       : e4b only (smoke), express-todo scenario
#   matrix=html        : every architecture, 2-turn html scenario —
#                        turn 1 creates mlx.html, turn 2 adds JS; scored on
#                        file existence/structure/content/JS plus the
#                        audit_format markers (junk filenames, tag leaks,
#                        thinking separation in the pi session)
#   matrix=html-quick  : e4b only, html scenario
#   PI_CASES=csv       : filter cases by label (e.g. PI_CASES=html-12b)
#   MLX_BIN=path       : server binary override (default: app bundle binary)
#
# Writes per-run logs into tests/pi-results/ and appends a
# summary line to tests/pi_integration_run.summary.tsv.

set -u
REPO="/Users/david/projects/agents/mlx-serve"
RESULTS="$REPO/tests/pi-results"
SUMMARY="$REPO/tests/pi_integration_run.summary.tsv"
MLX_BIN="${MLX_BIN:-$REPO/app/MLX Core.app/Contents/MacOS/mlx-serve}"
PI_MODELS_JSON="$HOME/.pi/agent/models.json"
PORT="${PORT:-8080}"
SERVED_MODEL="${SERVED_MODEL:-}"
WORKSPACE_ROOT="/tmp/pi_mlx_workspaces"

MATRIX="${1:-all}"
case "$MATRIX" in
    html*) SCENARIO="html" ;;
    *)     SCENARIO="todo" ;;
esac

mkdir -p "$RESULTS" "$WORKSPACE_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

# -----------------------------------------------------------------------------
# Model catalog. Each row: model_id | path | served_name | reasoning_supported
#   served_name is what mlx-serve reports in /v1/models (used by pi as id).
# -----------------------------------------------------------------------------
MODELS=(
    "e4b|$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit|gemma4|no"
    "a4b|$HOME/.mlx-serve/models/mlx-community/gemma-4-26b-a4b-it-4bit|gemma4|no"
    "qwen|$HOME/.mlx-serve/models/mlx-community/Qwen3.6-35B-A3B-6bit|qwen3_5_moe|yes"
)

kill_mlx_serve() {
    # Match by --port, not by binary path — MLX_BIN may point at the app
    # bundle OR a dev build; a path-based pattern silently leaves the other
    # one running and the next case reuses the wrong model.
    pkill -f "mlx-serve.*--serve.*--port $PORT" 2>/dev/null || true
    for _ in $(seq 1 10); do
        if ! pgrep -f "mlx-serve.*--serve.*--port $PORT" >/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

start_mlx_serve() {
    local path="$1"
    local logfile="$2"
    kill_mlx_serve
    "$MLX_BIN" --model "$path" --serve --port "$PORT" \
        --log-level info --ctx-size 32768 > "$logfile" 2>&1 &
    local pid=$!
    echo "$pid"
    # Wait up to 4 min for large models (35B has 27GB weights)
    for i in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "ready-after=${i}s" >&2
            return 0
        fi
        sleep 1
    done
    echo "FAILED to start mlx-serve" >&2
    return 1
}

write_pi_models_config() {
    local model_id="$1"
    local thinking_format="$2"   # "qwen" for Qwen, empty otherwise
    local reasoning="$3"          # true/false
    mkdir -p "$(dirname "$PI_MODELS_JSON")"
    if [ -n "$thinking_format" ]; then
        cat > "$PI_MODELS_JSON" <<EOF
{
  "providers": {
    "mlx": {
      "baseUrl": "http://127.0.0.1:$PORT/v1",
      "api": "openai-completions",
      "apiKey": "mlx",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "$thinking_format"
      },
      "models": [
        {"id": "$model_id", "name": "mlx-$model_id", "input": ["text"],
         "contextWindow": 32768, "maxTokens": 8192, "reasoning": $reasoning}
      ]
    }
  }
}
EOF
    else
        cat > "$PI_MODELS_JSON" <<EOF
{
  "providers": {
    "mlx": {
      "baseUrl": "http://127.0.0.1:$PORT/v1",
      "api": "openai-completions",
      "apiKey": "mlx",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {"id": "$model_id", "name": "mlx-$model_id", "input": ["text"],
         "contextWindow": 32768, "maxTokens": 8192, "reasoning": $reasoning}
      ]
    }
  }
}
EOF
    fi
}

# -----------------------------------------------------------------------------
# The actual agentic test. Two turns in a single pi session:
#   1. "build me a todo app with express"
#   2. "now add jest tests for it"
# Success criteria:
#   * package.json exists with express + jest deps
#   * at least one .js source file under project root that imports express
#   * at least one *.test.js file
#   * running `node -e "require('./...')"` doesn't crash on syntax
# -----------------------------------------------------------------------------
run_agent_turn() {
    # $1=model_id, $2=thinking_flag, $3=session_path, $4=workspace, $5=prompt, $6=logfile
    local model="$1" thinking="$2" session="$3" workspace="$4" prompt="$5" logfile="$6"
    local session_arg=""
    if [ -f "$session" ]; then session_arg="--session $session"; fi
    # Absolute timeout 15 min so pathological runs can't hang CI
    local t0=$(date +%s)
    (
        cd "$workspace"
        # PI_OFFLINE avoids network checks at startup
        PI_OFFLINE=1 pi --provider mlx --model "$model" \
            $thinking $session_arg --session-dir "$workspace/.pi-session" \
            --tools read,bash,edit,write,grep,find,ls \
            -p "$prompt" 2>&1 | tee -a "$logfile"
    )
    local rc=$?
    local t1=$(date +%s)
    echo "[elapsed=$((t1-t0))s exit=$rc]" | tee -a "$logfile"
    return $rc
}

score_workspace() {
    local ws="$1"
    local score=0
    local notes=""
    if [ -f "$ws/package.json" ]; then
        if grep -q '"express"' "$ws/package.json"; then
            score=$((score+1)); notes="$notes +package.json(express)"
        else
            notes="$notes -package.json(no-express)"
        fi
        if grep -qE '"jest"|"vitest"|"mocha"|"node:test"' "$ws/package.json"; then
            score=$((score+1)); notes="$notes +test-runner"
        fi
    else
        notes="$notes -no-package.json"
    fi
    local src
    src=$(find "$ws" -maxdepth 3 -name "*.js" -not -path "*/node_modules/*" -not -path "*/.pi-session/*" 2>/dev/null | grep -v '\.test\.js$' | head -1)
    if [ -n "$src" ] && grep -q "express" "$src" 2>/dev/null; then
        score=$((score+1)); notes="$notes +src($(basename $src))"
    else
        notes="$notes -no-express-src"
    fi
    local test
    test=$(find "$ws" -maxdepth 3 -name "*.test.js" -not -path "*/node_modules/*" -not -path "*/.pi-session/*" 2>/dev/null | head -1)
    if [ -n "$test" ]; then
        score=$((score+1)); notes="$notes +test($(basename $test))"
    else
        notes="$notes -no-test-file"
    fi
    # Bonus: does `npm test` actually pass? (skip if deps not installed)
    if [ -d "$ws/node_modules" ] && [ -n "$test" ]; then
        if (cd "$ws" && npx --no-install jest --silent >/dev/null 2>&1); then
            score=$((score+1)); notes="$notes +jest-green"
        else
            notes="$notes -jest-failed-or-not-run"
        fi
    else
        notes="$notes -no-node-modules"
    fi
    echo "$score|$notes"
}

# HTML scenario scorer — 2-turn create→extend flow:
#   turn 1 must produce mlx.html with real HTML structure and MLX content,
#   turn 2 must add the requested inline JavaScript to the SAME file.
score_workspace_html() {
    local ws="$1"
    local score=0
    local notes=""
    if [ -f "$ws/mlx.html" ]; then
        score=$((score+1)); notes="$notes +mlx.html"
        if grep -qi "<html" "$ws/mlx.html"; then
            score=$((score+1)); notes="$notes +html-structure"
        else
            notes="$notes -no-html-tag"
        fi
        if grep -qi "mlx" "$ws/mlx.html"; then
            score=$((score+1)); notes="$notes +mlx-content"
        else
            notes="$notes -no-mlx-content"
        fi
        # Inline JS counts as either a <script> block or an inline handler
        # (onclick=…) — both satisfy "inline JavaScript in the same file".
        if grep -qi "alert" "$ws/mlx.html" && grep -qiE "<script|onclick" "$ws/mlx.html"; then
            score=$((score+1)); notes="$notes +js-added"
        else
            notes="$notes -no-js"
        fi
    else
        notes="$notes -no-mlx.html"
    fi
    echo "$score|$notes"
}

# Format-correctness audit (Layer 3 of the cross-model format suite; the other
# layers are src/format_corpus_test.zig and tests/test_format_matrix.sh).
# Appends markers to the TSV notes column — score is untouched, but any `-`
# marker here is a hard-fail signal:
#   -junk-filename(<name>)  a created file whose basename carries tag/quote/
#                           brace garbage (the brace-swallow bug literally
#                           wrote "mlx_pi1.html`}" to disk)
#   +exact-filenames        the task's required files exist under their exact
#                           names (package.json, app.js, app.test.js)
#   -tag-leak(<match>)      a raw control tag (think/channel/tool-call/string
#                           delimiter) reached the agent transcript
audit_format() {
    local ws="$1" agent_log="$2"
    local notes=""
    local junk
    junk=$(find "$ws" -maxdepth 3 -not -path "*/node_modules/*" -not -path "*/.pi-session/*" -print0 2>/dev/null \
        | python3 -c '
import sys, os, re
bad = []
for p in sys.stdin.buffer.read().split(b"\0"):
    if not p:
        continue
    name = os.path.basename(p.decode("utf-8", "replace"))
    if re.search(r"[`<>|}{\"\x27]", name) or "<|" in name:
        bad.append(name)
print(",".join(bad[:3]))
')
    if [ -n "$junk" ]; then
        notes="$notes -junk-filename($junk)"
    elif { [ -f "$ws/package.json" ] && [ -f "$ws/app.js" ] && [ -f "$ws/app.test.js" ]; } \
        || [ -f "$ws/mlx.html" ]; then
        notes="$notes +exact-filenames"
    fi
    local leak
    leak=$(grep -aoE '</?think>|<\|channel>|<channel\|>|<\|tool_call|<tool_call>|<\|"\|>' "$agent_log" 2>/dev/null | head -1)
    if [ -n "$leak" ]; then
        notes="$notes -tag-leak($leak)"
    else
        notes="$notes +no-tag-leak"
    fi
    # Session-level audit: in the pi session jsonl, thinking must live in
    # thinking blocks and text blocks must never carry raw control tags.
    #   +thinking-blocks         at least one non-empty thinking block
    #   -thinking-in-text(<tag>) a raw control tag inside a TEXT block
    local sess
    sess=$(ls -t "$ws/.pi-session/"*.jsonl 2>/dev/null | head -1)
    if [ -n "$sess" ]; then
        local sres
        sres=$(python3 - "$sess" <<'PY'
import json, sys
tags = ["<think>", "</think>", "<|channel>", "<channel|>", "<|tool_call", "<tool_call>", '<|"|>']
think = 0
leak = ""
for line in open(sys.argv[1]):
    try:
        o = json.loads(line)
    except Exception:
        continue
    m = o.get("message") or {}
    if m.get("role") != "assistant":
        continue
    c = m.get("content")
    if not isinstance(c, list):
        continue
    for b in c:
        if b.get("type") == "thinking" and (b.get("thinking") or "").strip():
            think = 1
        if b.get("type") == "text":
            t = b.get("text", "")
            for tag in tags:
                if tag in t:
                    leak = tag
                    break
print(f"{think}|{leak}")
PY
)
        local sthink="${sres%%|*}" sleak="${sres#*|}"
        [ -n "$sleak" ] && notes="$notes -thinking-in-text($sleak)"
        [ "$sthink" = "1" ] && notes="$notes +thinking-blocks"
    fi
    echo "$notes"
}

run_one_case() {
    local label="$1"        # e.g. "qwen-think-medium"
    local path="$2"
    local served_name="$3"
    local thinking_flag="$4"
    local thinking_format="$5"
    local reasoning="$6"

    echo -e "${YELLOW}===== $label =====${NC}"
    local ws="$WORKSPACE_ROOT/$label"
    rm -rf "$ws" && mkdir -p "$ws"
    local server_log="$RESULTS/$label.server.log"
    local agent_log="$RESULTS/$label.agent.log"
    : > "$server_log"; : > "$agent_log"

    local pid load_start load_end
    load_start=$(date +%s)
    if [ -z "${SKIP_SERVER_START:-}" ]; then
        pid=$(start_mlx_serve "$path" "$server_log" 2> >(tee -a "$agent_log"))
        if [ -z "$pid" ]; then
            echo "FAIL: server failed" | tee -a "$agent_log"
            printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$label" "server-start-fail" "0" "" >> "$SUMMARY"
            return 1
        fi
        load_end=$(date +%s)
        echo "mlx-serve PID=$pid (loaded in $((load_end-load_start))s)" | tee -a "$agent_log"
    else
        pid="external"
        load_end=$load_start
        echo "[using already-running server at :$PORT]" | tee -a "$agent_log"
    fi

    # Allow env override of the model id used in pi's models.json.
    local effective_name="${SERVED_MODEL:-$served_name}"
    write_pi_models_config "$effective_name" "$thinking_format" "$reasoning"
    served_name="$effective_name"

    # Turn 1
    local turn1_prompt turn2_prompt_html
    if [ "$SCENARIO" = "html" ]; then
        turn1_prompt="Make me an html page about the MLX framework on Mac. Keep it minimal: a heading and a few bullet points. Save it as mlx.html in this directory. When done, say 'page ready'."
    else
        turn1_prompt="Create a minimal Express.js todo app in this directory. Requirements: package.json with express as dep, a file app.js exporting the Express app, in-memory todo storage, REST endpoints GET /todos, POST /todos (json body {text}), DELETE /todos/:id. Keep it in one file. Do NOT start the server yourself. When done, say 'app ready'."
    fi
    local t0=$(date +%s)
    run_agent_turn "$served_name" "$thinking_flag" "$ws/.pi-session/session.jsonl" \
        "$ws" "$turn1_prompt" "$agent_log"
    local t1=$(date +%s)
    echo "[turn1 elapsed=$((t1-t0))s]" | tee -a "$agent_log"

    # Turn 2 — continue same session
    local turn2_prompt
    if [ "$SCENARIO" = "html" ]; then
        turn2_prompt="Now add a button to mlx.html that shows an alert saying 'Hello from MLX' when clicked, using inline JavaScript in the same file. When done, say 'js added'."
    else
        turn2_prompt="Now add jest as a dev dependency in package.json and write a jest test file app.test.js that uses supertest to test all three endpoints. Also install dependencies with npm install. When done, run the tests and show me the output."
    fi
    local session_path="$ws/.pi-session/session.jsonl"
    # pi keeps the first session we started — use --continue
    t0=$(date +%s)
    (
        cd "$ws"
        PI_OFFLINE=1 pi --provider mlx --model "$served_name" \
            $thinking_flag --session-dir "$ws/.pi-session" --continue \
            --tools read,bash,edit,write,grep,find,ls \
            -p "$turn2_prompt" 2>&1 | tee -a "$agent_log"
    )
    t1=$(date +%s)
    echo "[turn2 elapsed=$((t1-t0))s]" | tee -a "$agent_log"

    # Grade
    local result max_score
    if [ "$SCENARIO" = "html" ]; then
        result=$(score_workspace_html "$ws")
        max_score=4
    else
        result=$(score_workspace "$ws")
        max_score=5
    fi
    local score=$(echo "$result" | cut -d'|' -f1)
    local notes=$(echo "$result" | cut -d'|' -f2)
    notes="$notes$(audit_format "$ws" "$agent_log")"
    local total_elapsed=$(( $(date +%s) - load_start ))
    echo "SCORE: $score/$max_score $notes [total=${total_elapsed}s]" | tee -a "$agent_log"
    printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$label" "$score/$max_score" "$total_elapsed" "$notes" >> "$SUMMARY"

    if [ -z "${SKIP_SERVER_START:-}" ]; then
        kill_mlx_serve
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Cases
# -----------------------------------------------------------------------------
declare -a CASES

# Case format: label|path|served_name|thinking_flag|thinking_format|reasoning  (6 fields, 5 pipes)
# Gemma 4 E4B — streaming, no thinking
CASES+=("e4b-stream|$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit|gemma4|||false")

if [ "$MATRIX" = "all" ]; then
    # Gemma 4 26B-A4B — streaming, no thinking
    CASES+=("a4b-stream|$HOME/.mlx-serve/models/mlx-community/gemma-4-26b-a4b-it-4bit|gemma4|||false")

    # Qwen3.6 35B — streaming, thinking OFF (pi sends enable_thinking=false)
    CASES+=("qwen-no-think|$HOME/.mlx-serve/models/mlx-community/Qwen3.6-35B-A3B-6bit|qwen3_5_moe|--thinking off|qwen|true")

    # Qwen3.6 35B — streaming, thinking MEDIUM (pi sends enable_thinking=true)
    CASES+=("qwen-think|$HOME/.mlx-serve/models/mlx-community/Qwen3.6-35B-A3B-6bit|qwen3_5_moe|--thinking medium|qwen|true")
fi

# HTML scenario: one case per supported architecture (gemma4 dense E4B + 12B,
# gemma4 MoE, qwen3_5_moe with thinking, qwen3_moe coder, gemma3 fallback).
if [ "$SCENARIO" = "html" ]; then
    CASES=()
    CASES+=("html-e4b|$HOME/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit|gemma4|||false")
    if [ "$MATRIX" = "html" ]; then
        CASES+=("html-12b|$HOME/.mlx-serve/models/mlx-community/gemma-4-12b-it-4bit|gemma4|||false")
        CASES+=("html-a4b|$HOME/.mlx-serve/models/mlx-community/gemma-4-26b-a4b-it-4bit|gemma4|||false")
        CASES+=("html-qwen-think|$HOME/.mlx-serve/models/mlx-community/Qwen3.6-35B-A3B-6bit|qwen3_5_moe|--thinking medium|qwen|true")
        CASES+=("html-qwen36|$HOME/.lmstudio/models/mlx-community/Qwen3.6-27B-4bit|qwen3_5|--thinking medium|qwen|true")
        CASES+=("html-coder|$HOME/.mlx-serve/models/mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit|qwen3_moe|||false")
        CASES+=("html-gemma3|$HOME/.mlx-serve/models/mlx-community/gemma-3-12b-it-qat-4bit|gemma3|||false")
        CASES+=("html-e4b-gguf|$HOME/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf|gguf|||false")
        CASES+=("html-ds4|$HOME/.mlx-serve/models/antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf|deepseek-v4|||false")
    fi
fi

if [ ! -f "$SUMMARY" ]; then
    printf "timestamp\tlabel\tscore\telapsed_s\tnotes\n" > "$SUMMARY"
fi

for case in "${CASES[@]}"; do
    IFS='|' read -r label path served_name thinking_flag thinking_format reasoning <<< "$case"
    if [ -n "${PI_CASES:-}" ]; then
        case ",$PI_CASES," in
            *",$label,"*) ;;
            *) continue ;;
        esac
    fi
    if [ ! -e "$path" ]; then
        echo -e "${YELLOW}SKIP${NC}: $label (model path missing: $path)"
        continue
    fi
    run_one_case "$label" "$path" "$served_name" "$thinking_flag" "$thinking_format" "$reasoning"
done

kill_mlx_serve
rm -f "$PI_MODELS_JSON"
echo -e "${GREEN}Done. Summary: $SUMMARY${NC}"
cat "$SUMMARY"
