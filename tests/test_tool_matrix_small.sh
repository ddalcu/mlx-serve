#!/bin/bash
# test_tool_matrix_small.sh — cross-model tool-call matrix for the sub-4B fleet.
#
# Companion to test_format_matrix.sh (which keys on format-family reps and only
# byte-checks a TRIVIAL write). This one boots the SMALL models one at a time and
# stresses the thing small models actually fail at: writing a file in one shot
# whose content carries the escaping hazards that strict JSON rejects — raw
# newlines and unescaped inner quotes. The bug those triggered (pre-fix) was the
# WHOLE tool call getting dropped and the file leaking into visible content; the
# server-side looseRepairToolCallJson recovery is what these assertions pin live.
#
# Per model, per request the battery asserts:
#   • bash baseline  — a trivial tool call fires at all (name + valid JSON args)
#   • quotes+newlines write — call FIRES (not dropped), args valid JSON, path
#     byte-exact, content carries the attribute-quote token `class="hero"`, and
#     NO <tool_call>/<|tool_call> tag leaks into visible content
#   • big-file write — a full HTML page: call fires, valid JSON, content length
#     over a floor (the call neither dropped nor silently emptied)
# each in BOTH non-stream and stream mode. It also greps the server log for the
# `[tool-parse] loose-repair recovered …` line and REPORTS per model whether the
# recovery actually engaged — i.e. which of these small models mis-escape in the
# wild (informational, never a failure).
#
# Outcome-based on purpose: a model that escapes correctly OR whose mangled
# output the server recovers both PASS — that is the cross-model "tool calls work"
# signal. A capability miss (a 0.8B that won't follow "exactly") shows as a FAIL
# for that one check on that one model, which is itself the signal you want.
# The deterministic proof of the recovery logic lives in the hermetic corpus
# test: `zig build test -Dtest-filter="format corpus"`.
#
# Usage: ./tests/test_tool_matrix_small.sh
#   TOOL_MODELS=qwen35-2b,gemma4-e4b   — csv filter of logical names
#   PORT=11298                          — override port
#   Missing model paths skip cleanly (exit 0 if nothing ran, nothing failed).
#
# Runtime: ~1–2 min/model (these are small), ~5–10 min for all five.

set -u
cd "$(dirname "$0")/.."

PORT="${PORT:-11298}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0; FAIL=0; MODEL_FAIL=0; RAN=0

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[1;34m'; NC='\033[0m'

# logical|display|path
MODELS=(
    "qwen35-0.8b|Qwen3.5 0.8B 4bit|$HOME/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"
    "qwen35-2b|Qwen3.5 2B 4bit|$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-2B-MLX-4bit"
    "qwen35-4b|Qwen3.5 4B 4bit|$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-4B-MLX-4bit"
    "gemma4-e2b|Gemma 4 E2B it 4bit|$HOME/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit"
    "gemma4-e4b|Gemma 4 E4B it 4bit|$HOME/.lmstudio/models/mlx-community/gemma-4-e4b-it-4bit"
)

if [ -n "${TOOL_MODELS:-}" ]; then
    IFS=',' read -r -a WANTED <<< "$TOOL_MODELS"
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        name="${entry%%|*}"
        for w in "${WANTED[@]}"; do [ "$name" = "$w" ] && FILTERED+=("$entry"); done
    done
    if [ ${#FILTERED[@]} -eq 0 ]; then
        echo "SKIP: TOOL_MODELS='$TOOL_MODELS' matched no known logical names"; exit 0
    fi
    MODELS=("${FILTERED[@]}")
fi

if [ ! -x "$BINARY" ]; then
    echo "FAIL: $BINARY not found — build first: zig build -Doptimize=ReleaseFast"; exit 1
fi

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else FAIL=$((FAIL+1)); MODEL_FAIL=$((MODEL_FAIL+1)); echo -e "  ${RED}FAIL${NC} $desc"; fi
}

jget() { python3 -c "import json,sys; r=json.loads(sys.argv[1]); print($2)" "$1" 2>/dev/null; }

# Tool definitions (path+content write tool; cmd bash tool).
BASH_TOOL='{"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}'
WRITE_TOOL='{"type":"function","function":{"name":"write_file","description":"Write a file to disk","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string","description":"Exact file content"}},"required":["path","content"]}}}'

# The quotes+newlines target the model must reproduce verbatim. Attribute quotes
# + inner quotes + an ampersand + multiple lines = the strict-JSON hazard set.
read -r -d '' QUOTES_TARGET <<'EOF'
<section class="hero">
  <a href="/go">Tom & "Jerry"</a>
</section>
EOF
QUOTES_MSG="Use the write_file tool RIGHT NOW to create a file named exactly snippet.html whose content is EXACTLY the following (copy it verbatim, no changes, no commentary):
$QUOTES_TARGET"

BIG_MSG='Use the write_file tool RIGHT NOW to create a complete, standalone HTML page saved as mars.html about the planet Mars. Include a full <!DOCTYPE html>, a <head> with <meta charset="UTF-8"> and a <title>, an embedded <style> block, an <h1>, and at least four <p> paragraphs. Call the tool now; do not ask questions.'

# Build a chat request body in python so the harness itself never has a quoting
# bug (json.dumps escapes the multiline/quote content correctly).
# args: message tools(or "-") stream(0/1) max_tokens
mkreq() {
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import json,sys
msg, tools, stream, maxtok = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body={"model":"x","messages":[{"role":"user","content":msg}],"max_tokens":int(maxtok),"temperature":0}
if tools!="-": body["tools"]=json.loads(tools)
if stream=="1": body["stream"]=True
print(json.dumps(body))
PY
}

# Accumulate an SSE stream into {"finish","content","calls":[{name,args}]}.
SSE_ACCUM='
import json,sys
calls={}; fr=None; c=""
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
    for tc in d.get("tool_calls") or []:
        i=tc.get("index",0); e=calls.setdefault(i,{"name":"","args":""})
        f=tc.get("function",{})
        if f.get("name"): e["name"]=f["name"]
        e["args"]+=f.get("arguments") or ""
print(json.dumps({"finish":fr,"content":c,"calls":[calls[k] for k in sorted(calls)]}))
'

# Verdict for a write call. argv: expected_path must_contain min_len
# stdin: {"name","args","content"}  (content = the assistant visible text)
# prints: call_ok|json_ok|path_ok|content_ok|leak_ok
WRITE_VERDICT='
import json,sys
want_path, must, minlen = sys.argv[1], sys.argv[2], int(sys.argv[3])
o=json.load(sys.stdin)
name=(o.get("name") or ""); args_s=o.get("args") or ""; vis=o.get("content") or ""
call_ok=1 if name in ("write_file","write","writeFile") else 0
json_ok=path_ok=content_ok=0
try:
    a=json.loads(args_s)
    if isinstance(a,dict):
        json_ok=1
        if a.get("path")==want_path: path_ok=1
        ct=a.get("content") or ""
        if (must=="-" or must in ct) and len(ct)>=minlen: content_ok=1
except Exception: pass
tags=["<tool_call",">tool_call","<|tool_call","<|channel","<channel|"]
leak_ok=0 if any(t in vis for t in tags) else 1
print(f"{call_ok}|{json_ok}|{path_ok}|{content_ok}|{leak_ok}")
'

run_model() {
    local logical="$1" display="$2" path="$3"
    echo -e "${BLUE}=== [$logical] $display ===${NC}"
    if [ ! -d "$path" ]; then echo -e "${YELLOW}SKIP${NC}: model dir not found: $path"; return 0; fi

    local log="/tmp/test_tool_matrix_$logical.log"
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 1
    "$BINARY" --model "$path" --serve --port "$PORT" --ctx-size 8192 --log-level debug > "$log" 2>&1 &
    local sp=$!
    local up=0
    for _ in $(seq 1 90); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$sp" 2>/dev/null || break
        sleep 2
    done
    if [ "$up" != "1" ]; then
        check "[$logical] server boots" 0
        tail -10 "$log" | sed 's/^/    /'; kill "$sp" 2>/dev/null; return 1
    fi
    RAN=$((RAN+1)); MODEL_FAIL=0

    local R ACC V

    # ── baseline bash tool call (non-stream) ──
    R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(mkreq 'Run the bash tool to print the current directory with: pwd' "[$BASH_TOOL]" 0 300)")
    V=$(jget "$R" "((r['choices'][0]['message'].get('tool_calls') or [{}])[0].get('function',{}) or {}).get('name','')")
    check "[$logical] bash tool call fires (name=bash)" "$([ "$V" = "bash" ] && echo 1 || echo 0)"

    # ── quotes+newlines write, NON-STREAM ──
    R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(mkreq "$QUOTES_MSG" "[$WRITE_TOOL]" 0 800)")
    V=$(jget "$R" "__import__('json').dumps({'name':(r['choices'][0]['message'].get('tool_calls') or [{}])[0].get('function',{}).get('name',''),'args':(r['choices'][0]['message'].get('tool_calls') or [{}])[0].get('function',{}).get('arguments',''),'content':r['choices'][0]['message'].get('content') or ''})" \
        | python3 -c "$WRITE_VERDICT" snippet.html 'class="hero"' 30)
    check "[$logical] quotes write (non-stream): call FIRES (not dropped)" "$([ "$(echo "$V"|cut -d'|' -f1)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] quotes write (non-stream): args valid JSON"         "$([ "$(echo "$V"|cut -d'|' -f2)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] quotes write (non-stream): path byte-exact"          "$([ "$(echo "$V"|cut -d'|' -f3)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] quotes write (non-stream): content keeps class=\"hero\"" "$([ "$(echo "$V"|cut -d'|' -f4)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] quotes write (non-stream): no tag leak in content"   "$([ "$(echo "$V"|cut -d'|' -f5)" = 1 ] && echo 1 || echo 0)"

    # ── quotes+newlines write, STREAM ──
    ACC=$(curl -sN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(mkreq "$QUOTES_MSG" "[$WRITE_TOOL]" 1 800)" | python3 -c "$SSE_ACCUM")
    V=$(jget "$ACC" "__import__('json').dumps({**((r['calls'] or [{'name':'','args':''}])[0]),'content':r['content']})" \
        | python3 -c "$WRITE_VERDICT" snippet.html 'class="hero"' 30)
    check "[$logical] quotes write (stream): call FIRES (not dropped)" "$([ "$(echo "$V"|cut -d'|' -f1)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] quotes write (stream): args valid JSON"         "$([ "$(echo "$V"|cut -d'|' -f2)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] quotes write (stream): content keeps class=\"hero\"" "$([ "$(echo "$V"|cut -d'|' -f4)" = 1 ] && echo 1 || echo 0)"

    # ── big-file write, NON-STREAM ──
    R=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(mkreq "$BIG_MSG" "[$WRITE_TOOL]" 0 3500)")
    V=$(jget "$R" "__import__('json').dumps({'name':(r['choices'][0]['message'].get('tool_calls') or [{}])[0].get('function',{}).get('name',''),'args':(r['choices'][0]['message'].get('tool_calls') or [{}])[0].get('function',{}).get('arguments',''),'content':r['choices'][0]['message'].get('content') or ''})" \
        | python3 -c "$WRITE_VERDICT" mars.html '-' 400)
    check "[$logical] big-file write (non-stream): call FIRES (not dropped)" "$([ "$(echo "$V"|cut -d'|' -f1)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] big-file write (non-stream): args valid JSON"         "$([ "$(echo "$V"|cut -d'|' -f2)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] big-file write (non-stream): content >=400 bytes"     "$([ "$(echo "$V"|cut -d'|' -f4)" = 1 ] && echo 1 || echo 0)"

    # ── big-file write, STREAM ──
    ACC=$(curl -sN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(mkreq "$BIG_MSG" "[$WRITE_TOOL]" 1 3500)" | python3 -c "$SSE_ACCUM")
    V=$(jget "$ACC" "__import__('json').dumps({**((r['calls'] or [{'name':'','args':''}])[0]),'content':r['content']})" \
        | python3 -c "$WRITE_VERDICT" mars.html '-' 400)
    check "[$logical] big-file write (stream): call FIRES (not dropped)" "$([ "$(echo "$V"|cut -d'|' -f1)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] big-file write (stream): args valid JSON"         "$([ "$(echo "$V"|cut -d'|' -f2)" = 1 ] && echo 1 || echo 0)"
    check "[$logical] big-file write (stream): content >=400 bytes"     "$([ "$(echo "$V"|cut -d'|' -f4)" = 1 ] && echo 1 || echo 0)"

    # Informational: did the server-side recovery actually engage for this model?
    local recov
    recov=$(grep -c "loose-repair recovered" "$log" 2>/dev/null); recov=${recov:-0}
    if [ "$recov" -gt 0 ]; then
        echo -e "  ${YELLOW}info${NC} [$logical] loose-repair recovery engaged ${recov}× (this model mis-escaped — the fix saved the call)"
    else
        echo -e "  ${YELLOW}info${NC} [$logical] loose-repair did not engage (model escaped cleanly this run)"
    fi
    if [ "$MODEL_FAIL" -gt 0 ]; then
        echo -e "  ${YELLOW}--- raw model output dumps for [$logical] ---${NC}"
        grep -a "raw generated text before tool parse" "$log" | tail -6 | sed 's/^/  /'
        echo "  (full log: $log)"
    fi
    kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null; return 0
}

trap 'pkill -f "mlx-serve.*--port $PORT" 2>/dev/null' EXIT

for entry in "${MODELS[@]}"; do
    IFS='|' read -r logical display path <<< "$entry"
    run_model "$logical" "$display" "$path"
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null; sleep 2
done

echo
echo "Results: $PASS passed, $FAIL failed ($RAN models ran)"
[ "$RAN" -eq 0 ] && { echo "SKIP: no fleet models found on this machine"; exit 0; }
[ "$FAIL" -eq 0 ]
