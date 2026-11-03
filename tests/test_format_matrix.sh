#!/bin/bash
# test_format_matrix.sh — live format matrix: 7 format-family representatives (Qwen think tags, Gemma 4 channel tags,
# Qwen MoE raw JSON, Gemma 3 fallback, GGUF/llama.cpp, DSV4-Flash/ds4) x 7 checks, one model at a time on port 11297.
#
#   FORMAT_MODELS=qwen36,gemma4-e4b ./tests/test_format_matrix.sh   # csv of logical names; gemma4-e4b ~3 min
#
# Checks: thinking off/on/truncated split, stream + tools content routing, tool-argument byte fidelity (chat and
# /v1/messages tool_use, stream and not), omitted max_tokens not cut at the old 256 default. Missing models skip.
# On a FAIL the server's `raw generated text before tool parse` lines are echoed for harvesting into the hermetic
# layer, src/host/chat/format_corpus_test.zig (`zig build test -Dtest-filter="format corpus"`).

set -u
source "$(dirname "$0")/lib/server.sh"

cd "$(dirname "$0")/.."
source tests/_lib_models.sh

PORT="${PORT:-11297}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0
FAIL=0
MODEL_FAIL=0
RAN=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[1;34m'
NC='\033[0m'

# logical|display|path relative to a model root, comma-separated alternatives (tests/_lib_models.sh)|engine|has_thinking
MODELS=(
    "qwen36|Qwen 3.6 27B dense (think tags + raw-JSON tools)|mlx-community/Qwen3.6-27B-4bit|mlx|yes"
    "gemma4-12b|Gemma 4 12B (channel tags + custom string-delim tool args)|mlx-community/gemma-4-12b-it-4bit|mlx|yes"
    "gemma4-e4b|Gemma 4 E4B (standard gemma4)|mlx-community/gemma-4-e4b-it-4bit|mlx|no"
    "qwen3-coder|Qwen3-Coder 30B-A3B (qwen3_moe flat-brace quirks)|mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit|mlx|no"
    "gemma3-12b|Gemma 3 12B (fallback chat format)|mlx-community/gemma-3-12b-it-qat-4bit,mlx-community/gemma-3-12b-it-4bit|mlx|no"
    "e4b-gguf|Gemma 4 E4B GGUF (embedded llama.cpp engine)|lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf|gguf|no"
    "ds4-flash|DeepSeek-V4-Flash GGUF (embedded ds4 engine)|antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf|gguf|no"
    "qwen38-27b|Qwen 3.8 27B dense (think tags + XML tools)|ddalcu/Qwen3.8-27B-MLX-Serve-4bit,ddalcu/Qwen3.8-27B-MLX-Serve-iQ-MLX-3.8bpw|mlx|yes"
    "gemma4-26b-gguf|Gemma 4 26B-A4B GGUF (embedded llama.cpp engine)|gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf|gguf|no"
    "flashnext-gguf|Qwen 3.8 Flash Next GGUF (embedded ds4 engine)|antirez/qwen3.8-flash-next-gguf/Qwen3.8-Flash-Next-Q2.gguf|gguf|yes"
)

# FORMAT_MODELS=csv filter of logical names. Unknown names simply match
# nothing — an all-bogus filter is a clean SKIP, not a failure.
if [ -n "${FORMAT_MODELS:-}" ]; then
    IFS=',' read -r -a WANTED <<< "$FORMAT_MODELS"
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        name="${entry%%|*}"
        for w in "${WANTED[@]}"; do
            if [ "$name" = "$w" ]; then FILTERED+=("$entry"); fi
        done
    done
    if [ ${#FILTERED[@]} -eq 0 ]; then
        echo "SKIP: FORMAT_MODELS='$FORMAT_MODELS' matched no known logical names"
        exit 0
    fi
    MODELS=("${FILTERED[@]}")
fi

if [ ! -x "$BINARY" ]; then
    echo "FAIL: $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); MODEL_FAIL=$((MODEL_FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

# no_leak <text> — 0 (clean) unless a control tag from any template family
# appears; prints the offending tag on leak.
no_leak() {
    python3 - "$1" <<'PY'
import sys
text = sys.argv[1]
tags = ["<think>", "</think>", "<|channel>", "<channel|>",
        "<|tool_call", "<tool_call", '<|"|>']
for t in tags:
    if t in text:
        print(f"  leaked tag: {t}", file=sys.stderr)
        sys.exit(1)
sys.exit(0)
PY
}

jget() { # jget <json> <python-expr over r>
    python3 -c "import json,sys; r=json.loads(sys.argv[1]); print($2)" "$1" 2>/dev/null
}

PROMPT='{"role":"user","content":"What is 17*23? Be brief."}'
BASH_TOOL='{"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}'
WRITE_TOOL='{"type":"function","function":{"name":"write","description":"Write a file to disk","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string","description":"File content"}},"required":["path","content"]}}}'
WRITE_PROMPT='{"role":"user","content":"Use the write tool to create a file named exactly report_v2.html containing <h1>Report</h1>. Do not ask questions, call the tool now."}'

# Accumulate an SSE stream into one JSON blob:
# {"finish":..., "content":..., "reasoning":..., "calls":[{"name","args"}...]}
SSE_ACCUM='
import json,sys
calls={}; fr=None; c=r=""
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data:"): continue
    p=line[5:].strip()
    if p=="[DONE]": break
    try: o=json.loads(p)
    except: continue
    ch=(o.get("choices") or [{}])[0]
    if ch.get("finish_reason"): fr=ch["finish_reason"]
    d=ch.get("delta",{})
    c+=d.get("content") or ""
    r+=d.get("reasoning_content") or ""
    for tc in d.get("tool_calls") or []:
        i=tc.get("index",0)
        e=calls.setdefault(i,{"name":"","args":""})
        f=tc.get("function",{})
        if f.get("name"): e["name"]=f["name"]
        e["args"]+=f.get("arguments") or ""
print(json.dumps({"finish":fr,"content":c,"reasoning":r,
                  "calls":[calls[k] for k in sorted(calls)]}))
'

# Validate a tool call: name == write, args valid JSON, path byte-exact.
# Input on stdin: {"name":..., "args":...}; prints "name_ok|json_ok|path_ok".
TOOL_VERDICT='
import json,sys
tc=json.load(sys.stdin)
name_ok=1 if tc.get("name")=="write" else 0
json_ok=path_ok=0
try:
    args=json.loads(tc.get("args") or "")
    json_ok=1 if isinstance(args,dict) else 0
    path_ok=1 if args.get("path")=="report_v2.html" else 0
except Exception: pass
print(f"{name_ok}|{json_ok}|{path_ok}")
'

run_model() {
    local logical="$1" display="$2" path="$3" engine="$4" has_thinking="$5"

    echo -e "${BLUE}=== [$logical] $display ===${NC}"

    local rel="$path" cands
    IFS=',' read -r -a cands <<< "$rel"
    if ! path=$(find_model "${cands[@]}"); then
        echo -e "${YELLOW}SKIP${NC}: $rel not on any model root"
        return 0
    fi
    if ! model_fits "$path"; then
        echo -e "${YELLOW}SKIP${NC}: $(model_gb "$path") GB > $(max_model_gb) GB budget"
        return 0
    fi

    local log="/tmp/test_format_matrix_$logical.log"
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    sleep 1
    "$BINARY" --model "$path" --serve --port "$PORT" --ctx-size 8192 \
        --log-level debug > "$log" 2>&1 &
    local sp=$!

    # MoE models can take 90+ s on cold weight pages — allow 240 s.
    local up=0
    srv_wait_health "$BASE" 120 2 "$sp" && up=1
    if [ "$up" != "1" ]; then
        check "[$logical] server boots" 0
        echo "  server log tail:"; tail -10 "$log" | sed 's/^/    /'
        kill "$sp" 2>/dev/null
        return 1
    fi
    RAN=$((RAN + 1))
    MODEL_FAIL=0

    # ── 1. Thinking OFF baseline ──
    local R CONTENT
    R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":200,\"temperature\":0}")
    CONTENT=$(jget "$R" "(r['choices'][0]['message'].get('content') or '')")
    check "[$logical] 1. thinking-off: answer (391) in content" \
        "$(echo "$CONTENT" | grep -q 391 && echo 1 || echo 0)"
    check "[$logical] 1. thinking-off: no control-tag leak" \
        "$(no_leak "$CONTENT" && echo 1 || echo 0)"

    if [ "$has_thinking" = "yes" ]; then
        # ── 2. Thinking ON, truncated (pins bug 1 at HTTP level) ──
        local REASONING
        R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":120,\"temperature\":0,\"enable_thinking\":true}")
        CONTENT=$(jget "$R" "(r['choices'][0]['message'].get('content') or '')")
        REASONING=$(jget "$R" "(r['choices'][0]['message'].get('reasoning_content') or '')")
        # A short thinker can close the thought inside the cap; only a cut thought must leave content empty.
        FIN=$(jget "$R" "r['choices'][0]['finish_reason']")
        check "[$logical] 2. truncated thinking: content empty when the cap cut the thought" \
            "$([ "$FIN" != "length" ] || [ -z "$CONTENT" ] && echo 1 || echo 0)"
        check "[$logical] 2. truncated thinking: reasoning_content non-empty" \
            "$([ -n "$REASONING" ] && echo 1 || echo 0)"

        # ── 3. Thinking ON, full round ──
        local RTOK
        R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":3000,\"temperature\":0,\"enable_thinking\":true}")
        CONTENT=$(jget "$R" "(r['choices'][0]['message'].get('content') or '')")
        REASONING=$(jget "$R" "(r['choices'][0]['message'].get('reasoning_content') or '')")
        RTOK=$(jget "$R" "r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0)")
        check "[$logical] 3. full round: answer (391) in content" \
            "$(echo "$CONTENT" | grep -q 391 && echo 1 || echo 0)"
        check "[$logical] 3. full round: no control-tag leak in content" \
            "$(no_leak "$CONTENT" && echo 1 || echo 0)"
        check "[$logical] 3. full round: reasoning separated" \
            "$([ -n "$REASONING" ] && echo 1 || echo 0)"
        check "[$logical] 3. full round: usage reasoning_tokens > 0" \
            "$([ "${RTOK:-0}" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
    fi

    # ── 4. Stream + tools (+thinking where supported): answer must arrive as
    #      CONTENT deltas (pins the answer-misfiled-as-reasoning stream bug) ──
    local THINK_FIELD=""
    [ "$has_thinking" = "yes" ] && THINK_FIELD=',"enable_thinking":true'
    local ACC
    ACC=$(curl -sN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":3000,\"temperature\":0,\"stream\":true,\"tools\":[$BASH_TOOL]$THINK_FIELD}" \
        | python3 -c "$SSE_ACCUM")
    CONTENT=$(jget "$ACC" "r['content']")
    check "[$logical] 4. stream+tools: answer (391) in CONTENT deltas" \
        "$(echo "$CONTENT" | grep -q 391 && echo 1 || echo 0)"
    check "[$logical] 4. stream+tools: no control-tag leak" \
        "$(no_leak "$CONTENT" && echo 1 || echo 0)"

    # ── 5. Tool-call fidelity, non-stream (pins the brace-swallow bug live) ──
    local VERDICT
    R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"x\",\"messages\":[$WRITE_PROMPT],\"max_tokens\":500,\"temperature\":0,\"tools\":[$WRITE_TOOL]}")
    VERDICT=$(jget "$R" "__import__('json').dumps({'name':(r['choices'][0]['message'].get('tool_calls') or [{}])[0].get('function',{}).get('name',''),'args':(r['choices'][0]['message'].get('tool_calls') or [{}])[0].get('function',{}).get('arguments','')})" \
        | python3 -c "$TOOL_VERDICT")
    check "[$logical] 5. non-stream tool call: name == write" \
        "$([ "${VERDICT%%|*}" = "1" ] && echo 1 || echo 0)"
    check "[$logical] 5. non-stream tool call: args are valid JSON" \
        "$(echo "$VERDICT" | cut -d'|' -f2 | grep -q 1 && echo 1 || echo 0)"
    check "[$logical] 5. non-stream tool call: path byte-exact report_v2.html" \
        "$([ "${VERDICT##*|}" = "1" ] && echo 1 || echo 0)"

    # ── 6. Tool-call fidelity, stream (accumulated delta.tool_calls) ──
    ACC=$(curl -sN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"x\",\"messages\":[$WRITE_PROMPT],\"max_tokens\":500,\"temperature\":0,\"stream\":true,\"tools\":[$WRITE_TOOL]}" \
        | python3 -c "$SSE_ACCUM")
    VERDICT=$(jget "$ACC" "__import__('json').dumps((r['calls'] or [{'name':'','args':''}])[0])" \
        | python3 -c "$TOOL_VERDICT")
    check "[$logical] 6. stream tool call: name == write" \
        "$([ "${VERDICT%%|*}" = "1" ] && echo 1 || echo 0)"
    check "[$logical] 6. stream tool call: args are valid JSON" \
        "$(echo "$VERDICT" | cut -d'|' -f2 | grep -q 1 && echo 1 || echo 0)"
    check "[$logical] 6. stream tool call: path byte-exact report_v2.html" \
        "$([ "${VERDICT##*|}" = "1" ] && echo 1 || echo 0)"

    # ── 7. Omitted max_tokens must not truncate at the old 256 default ──
    local FINISH CTOK
    R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"x","messages":[{"role":"user","content":"List the numbers from 1 to 200, one per line. No other text."}],"temperature":0}')
    FINISH=$(jget "$R" "r['choices'][0].get('finish_reason','')")
    CTOK=$(jget "$R" "r['usage'].get('completion_tokens',0)")
    check "[$logical] 7. omitted max_tokens: finish_reason != length (got $FINISH @ $CTOK tok)" \
        "$([ "$FINISH" != "length" ] && echo 1 || echo 0)"
    check "[$logical] 7. omitted max_tokens: completion_tokens != 256" \
        "$([ "${CTOK:-0}" != "256" ] && echo 1 || echo 0)"

    # ── 8. /v1/messages stream + tools (+thinking): the Claude Code surface.
    #      Tool call must arrive as a tool_use block (name + byte-exact path),
    #      and no text/thinking delta may carry a raw control tag — pins the
    #      mid-text re-opened thought channel leak observed live (2026-06-10).
    local ANTH_THINK=""
    [ "$has_thinking" = "yes" ] && ANTH_THINK=',"thinking":{"type":"enabled","budget_tokens":2000}'
    local AV
    AV=$(curl -sN "$BASE/v1/messages" -H 'Content-Type: application/json' \
        -d "{\"model\":\"x\",\"max_tokens\":2000,\"stream\":true,\"temperature\":0,\"tools\":[{\"name\":\"write\",\"description\":\"Write a file to disk\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}],\"messages\":[$WRITE_PROMPT]$ANTH_THINK}" \
        | python3 -c '
import json,sys
blocks={}
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data:"): continue
    try: ev=json.loads(line[5:])
    except: continue
    t=ev.get("type")
    if t=="content_block_start":
        cb=ev["content_block"]
        blocks[ev["index"]]={"type":cb["type"],"name":cb.get("name"),"text":"","json":""}
    elif t=="content_block_delta":
        d=ev["delta"]; b=blocks.get(ev["index"])
        if b is None: continue
        b["text"]+= (d.get("text","") or "")
        b["json"]+= d.get("partial_json","")
tool=next((b for b in blocks.values() if b["type"]=="tool_use"),None)
name_ok=json_ok=path_ok=0
if tool:
    name_ok=1 if tool["name"]=="write" else 0
    try:
        a=json.loads(tool["json"]); json_ok=isinstance(a,dict); path_ok=1 if a.get("path")=="report_v2.html" else 0
    except Exception: pass
tags=["<think>","</think>","<|channel>","<channel|>","<|tool_call","<tool_call","<|\"|>"]
text="".join(b["text"] for b in blocks.values() if b["type"]!="tool_use")
leak=next((t for t in tags if t in text),"")
print(f"{name_ok}|{int(json_ok)}|{path_ok}|{leak}")')
    check "[$logical] 8. /v1/messages stream: tool_use block name == write" \
        "$([ "$(echo "$AV" | cut -d'|' -f1)" = "1" ] && echo 1 || echo 0)"
    check "[$logical] 8. /v1/messages stream: args valid + path byte-exact" \
        "$([ "$(echo "$AV" | cut -d'|' -f2)" = "1" ] && [ "$(echo "$AV" | cut -d'|' -f3)" = "1" ] && echo 1 || echo 0)"
    check "[$logical] 8. /v1/messages stream: no control-tag leak in deltas" \
        "$([ -z "$(echo "$AV" | cut -d'|' -f4)" ] && echo 1 || echo 0)"

    # On any FAIL for this model, surface the raw-output dumps so a corpus
    # entry can be harvested straight from the run output.
    if [ "$MODEL_FAIL" -gt 0 ]; then
        echo -e "  ${YELLOW}--- raw model output dumps for [$logical] (corpus harvesting) ---${NC}"
        grep -a "raw generated text before tool parse" "$log" | tail -8 | sed 's/^/  /'
        echo "  (full log: $log)"
    fi

    kill "$sp" 2>/dev/null
    wait "$sp" 2>/dev/null
    return 0
}

trap 'pkill -f "mlx-serve.*--port $PORT" 2>/dev/null' EXIT

for entry in "${MODELS[@]}"; do
    IFS='|' read -r logical display path engine has_thinking <<< "$entry"
    run_model "$logical" "$display" "$path" "$engine" "$has_thinking"
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    sleep 2
done

echo
echo "Results: $PASS passed, $FAIL failed ($RAN models ran)"
if [ "$RAN" -eq 0 ]; then
    echo "SKIP: no matrix models found on this machine"
    exit 0
fi
[ "$FAIL" -eq 0 ]
