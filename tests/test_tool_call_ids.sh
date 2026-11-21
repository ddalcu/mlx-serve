#!/bin/bash
# test_tool_call_ids.sh — tool-call ids are random and long enough on every surface: a forced call on /v1/messages
# (twice, and streaming), chat completions (streaming and not) and /v1/responses returns an id with the route's prefix,
# at least 22 chars (Claude Code rejects shorter tool_use ids), no 13-digit millisecond stamp, and no two ids alike.
#
#   ./tests/test_tool_call_ids.sh [model_dir] [port]   # default Qwen3.5-4B
#
# Hermetic twin: `makeId: 144-bit CSPRNG body as base64url, no clock or counter`.

set -u
source "$(dirname "$0")/lib/server.sh"

source "$(dirname "$0")/_lib_models.sh"
MODEL="${1:-$(find_model mlx-community/Qwen3.5-4B-MLX-4bit lmstudio-community/Qwen3.5-4B-MLX-4bit mlx-community/Qwen3.5-0.8B-MLX-4bit)}"
PORT="${2:-11277}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0
FAIL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

check() {
    if [ "$2" = "1" ]; then PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1";
    else FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1"; fi
}

if [ ! -d "$MODEL" ]; then echo "skip: model not found ($MODEL)"; exit 0; fi

srv_preclean "$PORT"
LOG=$(mktemp); OUT=$(mktemp); IDS=$(mktemp)
"$BINARY" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'srv_stop $SRV; rm -f "$LOG" "$OUT" "$IDS"' EXIT
srv_wait_health "$BASE" 240 0.5 "$SRV" || { echo "server failed to start"; tail -5 "$LOG"; exit 1; }

MSG='[{"role":"user","content":"What is the weather in Paris?"}]'
SCHEMA='{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}'
DESC='Get the current weather for a city.'

# id_check <label> <prefix> <python expression over the parsed body or SSE events, yielding the ids>
id_check() {
    check "$1" "$(python3 -c '
import json,re,sys
raw=open(sys.argv[1]).read()
try: j=json.loads(raw)
except ValueError: j=None
ev=[json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ") and "[DONE]" not in l]
ids=eval(sys.argv[3])
ok=bool(ids) and all(i.startswith(sys.argv[2]) and len(i)>=22 and not re.search(r"\d{13}",i) for i in ids)
open(sys.argv[4],"a").write("".join(i+"\n" for i in ids))
print(1 if ok else 0)' "$OUT" "$2" "$3" "$IDS" 2>/dev/null)"
}

echo "[tool-ids] === $(basename "$MODEL") ==="

MBODY="{\"model\":\"x\",\"max_tokens\":512,\"temperature\":0,\"messages\":$MSG,\"tool_choice\":{\"type\":\"any\"},
  \"tools\":[{\"name\":\"get_weather\",\"description\":\"$DESC\",\"input_schema\":$SCHEMA}]"
for n in 1 2; do
    curl -s -m 300 "$BASE/v1/messages" -H 'content-type: application/json' -d "$MBODY}" > "$OUT"
    id_check "messages ($n): toolu_ id" toolu_ '[b["id"] for b in j["content"] if b["type"]=="tool_use"]'
done
curl -sN -m 300 "$BASE/v1/messages" -H 'content-type: application/json' -d "$MBODY,\"stream\":true}" > "$OUT"
id_check "messages, stream: toolu_ id" toolu_ \
    '[e["content_block"]["id"] for e in ev if e.get("type")=="content_block_start" and e["content_block"]["type"]=="tool_use"]'

CBODY="{\"model\":\"x\",\"max_tokens\":512,\"temperature\":0,\"messages\":$MSG,\"tool_choice\":\"required\",
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"$DESC\",\"parameters\":$SCHEMA}}]"
curl -s -m 300 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d "$CBODY}" > "$OUT"
id_check "chat: call_ id" call_ '[t["id"] for t in j["choices"][0]["message"].get("tool_calls") or []]'
curl -sN -m 300 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d "$CBODY,\"stream\":true}" > "$OUT"
id_check "chat, stream: call_ id" call_ \
    '[t["id"] for e in ev for c in e.get("choices") or [] for t in c["delta"].get("tool_calls") or [] if t.get("id")]'

curl -s -m 300 "$BASE/v1/responses" -H 'content-type: application/json' -d "{\"model\":\"x\",\"input\":\"What is the weather in Paris?\",
  \"tool_choice\":\"required\",\"temperature\":0,\"max_output_tokens\":512,
  \"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"description\":\"$DESC\",\"parameters\":$SCHEMA}]}" > "$OUT"
id_check "responses: call_ id" call_ '[o["call_id"] for o in j.get("output",[]) if o.get("type")=="function_call"]'

check "no two ids alike" "$([ -s "$IDS" ] && [ "$(sort "$IDS" | uniq -d | wc -l)" -eq 0 ] && echo 1 || echo 0)"

echo "[tool-ids] $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
