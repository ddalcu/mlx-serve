#!/usr/bin/env bash
# Ollama-compatible API (/api/*) end-to-end: version/tags/show/ps, chat
# (stream NDJSON + non-stream + tool calls + done_reason), generate
# (templated + raw), embed (new + legacy), name resolution ("qwen3.5:latest"
# style), pull fast-path (already on disk -> success, no network), and the
# `mlx-serve list` CLI. Complements the hermetic translation/sink tests in
# src/ollama.zig — this pins the routing + Conn-sink plumbing live.
#
# Usage: [OLLAMA_CHAT_MODEL=<dir>] [OLLAMA_EMBED_ID=<id>] ./tests/test_ollama_api.sh [port]
set -uo pipefail
PORT="${1:-11436}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
LOG=/tmp/test_ollama_api.log
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

CHAT="${OLLAMA_CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
[ -n "$CHAT" ] || { echo "SKIP: no chat model (set OLLAMA_CHAT_MODEL)"; exit 0; }
CHAT_ID="$(basename "$CHAT")"

"$BIN" --serve --model "$CHAT" --model-dir "$HOME/.mlx-serve/models" --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -8 "$LOG"; exit 1; }
  sleep 1
done

api() { curl -s -m 300 "http://127.0.0.1:$PORT$1" "${@:2}"; }
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

# ── 1. /api/version ──
V=$(api /api/version | python3 -c 'import sys,json;print(json.load(sys.stdin)["version"])' 2>/dev/null)
[ -n "$V" ] && ok "/api/version -> $V" || bad "/api/version"

# ── 2. /api/tags lists the loaded model with :latest name + details ──
api /api/tags > /tmp/ollama_tags.json
python3 - "$CHAT_ID" /tmp/ollama_tags.json <<'PY' && ok "/api/tags shape + :latest names" || bad "/api/tags"
import sys, json
want, path = sys.argv[1], sys.argv[2]
models = json.load(open(path))["models"]
assert models, "no models"
m = [x for x in models if want in x["name"]]
assert m, f"{want} not in tags: {[x['name'] for x in models]}"
e = m[0]
assert e["name"].endswith(":latest") and e["model"] == e["name"]
assert len(e["digest"]) == 64 and "family" in e["details"]
assert e["modified_at"][:2] == "20", "modified_at should be a real date"
PY

# ── 3. /api/show by unique basename ──
api /api/show -X POST -d "{\"model\":\"$CHAT_ID\"}" > /tmp/ollama_show.json
python3 - /tmp/ollama_show.json <<'PY' && ok "/api/show capabilities + model_info" || bad "/api/show"
import sys, json
d = json.load(open(sys.argv[1]))
assert "completion" in d["capabilities"], d["capabilities"]
assert d["model_info"]["general.architecture"], d["model_info"]
PY

# ── 4. /api/chat non-stream ──
api /api/chat -X POST -d '{"model":"mlx-serve","stream":false,"messages":[{"role":"user","content":"Say exactly: hello"}],"options":{"temperature":0,"num_predict":64}}' > /tmp/ollama_chat_ns.json
python3 - /tmp/ollama_chat_ns.json <<'PY' && ok "/api/chat non-stream content + stats" || bad "/api/chat non-stream ($(head -c 200 /tmp/ollama_chat_ns.json))"
import sys, json
d = json.load(open(sys.argv[1]))
assert d["done"] is True and d["message"]["role"] == "assistant"
assert len(d["message"]["content"]) > 0
assert d["eval_count"] > 0 and d["prompt_eval_count"] > 0 and d["total_duration"] > 0
PY

# ── 5. /api/chat streaming NDJSON: every line valid JSON, final has stats ──
api /api/chat -X POST -d '{"model":"mlx-serve","messages":[{"role":"user","content":"Count to five."}],"options":{"temperature":0,"num_predict":64}}' > /tmp/ollama_chat_stream.ndjson
python3 - /tmp/ollama_chat_stream.ndjson <<'PY' && ok "/api/chat stream NDJSON lifecycle" || bad "/api/chat stream"
import sys, json
lines = [l for l in open(sys.argv[1]).read().splitlines() if l.strip()]
assert len(lines) >= 2, f"expected multiple NDJSON lines, got {len(lines)}"
objs = [json.loads(l) for l in lines]          # every line must parse
assert all(not o["done"] for o in objs[:-1])
content = "".join(o["message"]["content"] for o in objs)
assert content, "no streamed content"
last = objs[-1]
assert last["done"] and last["done_reason"] in ("stop", "length")
assert last["eval_count"] > 0 and last["eval_duration"] > 0
assert "data:" not in "".join(lines), "SSE framing leaked into NDJSON"
PY

# ── 6. Ollama-style name resolution: short/tagged name routes (or falls back) ──
C=$(api /api/chat -X POST -d '{"model":"qwen3.5:latest","stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"temperature":0,"num_predict":16}}' | python3 -c 'import sys,json;d=json.load(sys.stdin);print(len(d["message"]["content"])>0)' 2>/dev/null)
[ "$C" = "True" ] && ok "tagged model name accepted (qwen3.5:latest)" || bad "tagged model name"

# ── 7. done_reason length via options.num_predict ──
R=$(api /api/chat -X POST -d '{"model":"mlx-serve","stream":false,"messages":[{"role":"user","content":"Count from 1 to 500, comma separated."}],"options":{"temperature":0,"num_predict":8}}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["done_reason"])' 2>/dev/null)
[ "$R" = "length" ] && ok "num_predict -> done_reason length" || bad "num_predict cap (got '$R')"

# ── 8. /api/chat tool calling: arguments must be an OBJECT ──
api /api/chat -X POST -d '{"model":"mlx-serve","stream":false,"messages":[{"role":"user","content":"What is the weather in Paris? You MUST call the get_weather function."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"options":{"temperature":0}}' > /tmp/ollama_tools.json
python3 - /tmp/ollama_tools.json <<'PY' && ok "/api/chat tool call with object arguments" || bad "/api/chat tools ($(head -c 200 /tmp/ollama_tools.json))"
import sys, json
d = json.load(open(sys.argv[1]))
tcs = d["message"].get("tool_calls")
assert tcs, f"no tool_calls: {json.dumps(d)[:300]}"
fn = tcs[0]["function"]
assert fn["name"] == "get_weather"
assert isinstance(fn["arguments"], dict), f"arguments must be an object, got {type(fn['arguments'])}"
PY

# ── 9. /api/generate templated (non-stream) ──
api /api/generate -X POST -d '{"model":"mlx-serve","stream":false,"prompt":"Say exactly: pong","options":{"temperature":0,"num_predict":32}}' > /tmp/ollama_gen.json
python3 - /tmp/ollama_gen.json <<'PY' && ok "/api/generate templated" || bad "/api/generate ($(head -c 200 /tmp/ollama_gen.json))"
import sys, json
d = json.load(open(sys.argv[1]))
assert d["done"] is True and len(d["response"]) > 0
assert d["context"] == []
PY

# ── 10. /api/generate raw streaming (FIM path, /v1/completions inside) ──
api /api/generate -X POST -d '{"model":"mlx-serve","raw":true,"prompt":"1, 2, 3, 4,","options":{"temperature":0,"num_predict":8}}' > /tmp/ollama_gen_raw.ndjson
python3 - /tmp/ollama_gen_raw.ndjson <<'PY' && ok "/api/generate raw stream" || bad "/api/generate raw"
import sys, json
lines = [json.loads(l) for l in open(sys.argv[1]).read().splitlines() if l.strip()]
assert lines and lines[-1]["done"]
assert "".join(o["response"] for o in lines), "no raw completion text"
PY

# ── 11. /api/embed + legacy /api/embeddings (needs the bge encoder on disk) ──
EMBED_ID="${OLLAMA_EMBED_ID:-bge-small-en-v1.5-8bit}"
if [ -d "$HOME/.mlx-serve/models/mlx-community/$EMBED_ID" ]; then
  api /api/embed -X POST -d "{\"model\":\"$EMBED_ID\",\"input\":[\"hello\",\"world\"]}" > /tmp/ollama_embed.json
  python3 - /tmp/ollama_embed.json <<'PY' && ok "/api/embed batch" || bad "/api/embed ($(head -c 200 /tmp/ollama_embed.json))"
import sys, json
d = json.load(open(sys.argv[1]))
assert len(d["embeddings"]) == 2 and len(d["embeddings"][0]) > 100
assert d["prompt_eval_count"] > 0
PY
  N=$(api /api/embeddings -X POST -d "{\"model\":\"$EMBED_ID\",\"prompt\":\"hello\"}" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["embedding"]))' 2>/dev/null)
  [ "${N:-0}" -gt 100 ] && ok "/api/embeddings legacy single vector (dim $N)" || bad "/api/embeddings legacy"
else
  echo "SKIP: embed checks (no $EMBED_ID on disk)"
fi

# ── 12. /api/ps shows resident models ──
P=$(api /api/ps | python3 -c 'import sys,json;m=json.load(sys.stdin)["models"];print(len([x for x in m if x["size_vram"]>0]))' 2>/dev/null)
[ "${P:-0}" -ge 1 ] && ok "/api/ps resident models ($P)" || bad "/api/ps"

# ── 13. /api/pull fast-path: model already on disk -> success, no network ──
api /api/pull -X POST -d '{"model":"qwen3.5:0.8b"}' > /tmp/ollama_pull.ndjson
grep -q '"status":"success' /tmp/ollama_pull.ndjson && ok "/api/pull already-present fast path" || bad "/api/pull ($(head -c 200 /tmp/ollama_pull.ndjson))"

# ── 14. unsupported registry-mutating endpoint -> explicit 501 ──
CODE=$(curl -s -o /tmp/ollama_del.json -w '%{http_code}' -m 30 -X POST "http://127.0.0.1:$PORT/api/delete" -d '{"model":"x"}')
[ "$CODE" = "501" ] && grep -q '"error"' /tmp/ollama_del.json && ok "/api/delete -> 501 + error body" || bad "/api/delete (code $CODE)"

# ── 15. errors carry the Ollama {"error": ...} shape ──
CODE=$(curl -s -o /tmp/ollama_err.json -w '%{http_code}' -m 30 -X POST "http://127.0.0.1:$PORT/api/chat" -d '{"model":"mlx-serve"}')
[ "$CODE" = "400" ] && python3 -c 'import json;assert isinstance(json.load(open("/tmp/ollama_err.json"))["error"],str)' 2>/dev/null && ok "missing messages -> 400 {error}" || bad "error shape (code $CODE)"

# ── 16. CLI: mlx-serve list shows the model ──
"$BIN" list | grep -q "Qwen3.5-0.8B" && ok "mlx-serve list" || bad "mlx-serve list"

# ── 17. (opt-in, network) real `mlx-serve pull` into a scratch HOME ──
if [ "${OLLAMA_PULL_NET:-0}" = "1" ]; then
  FAKEHOME=$(mktemp -d)
  HOME="$FAKEHOME" "$BIN" pull bge-small >/dev/null 2>&1 \
    && [ -s "$FAKEHOME/.mlx-serve/models/mlx-community/bge-small-en-v1.5-8bit/model.safetensors" ] \
    && ok "mlx-serve pull (real download)" || bad "mlx-serve pull network"
  rm -rf "$FAKEHOME"
else
  echo "SKIP: network pull (set OLLAMA_PULL_NET=1)"
fi

# ── 18. (opt-in, slow) `mlx-serve run` REPL round-trip through a pty ──
if [ "${OLLAMA_RUN_REPL:-0}" = "1" ]; then
  OUT=$( (printf 'Say exactly: pong\n'; sleep 40; printf '/bye\n'; sleep 3) | \
    script -q /dev/null "$BIN" run qwen3.5 --port $((PORT+1)) 2>&1 | tail -6 )
  echo "$OUT" | grep -q "chat is live" && echo "$OUT" | grep -qi "pong" \
    && ok "mlx-serve run REPL round-trip" || bad "mlx-serve run REPL ($OUT)"
else
  echo "SKIP: run REPL (set OLLAMA_RUN_REPL=1)"
fi

echo ""
echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" = "0" ] || exit 1
