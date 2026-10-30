#!/usr/bin/env bash
# test_ane_engine.sh — the Neural Engine engine over HTTP (src/engines/ane/lm.zig). `"ane": true` in a scratch
# model-settings.json serves a dense qwen3_5 pack on the ANE (`[ane] ready`, meta.engine ane, 64K context): stream ==
# non-stream bytes; logprobs are served; forced tool calls and images are 400s naming the Neural Engine; a ~15K-token prompt
# finds its needle across `context 4096 -> 16384` and its repeat restores a checkpoint. Without the setting the pack
# stays on the GPU; `--ane` on a pack the engine cannot run serves it on the GPU by name.
#
#   ./tests/test_ane_engine.sh [port]   # ANE_MODEL (Qwen3.5-0.8B), ANE_OTHER_MODEL (LFM2.5-2.6B)
#
# Keeps the real compile cache (MLX_SERVE_ANE_CACHE_DIR): only a Mac's first run compiles. SKIPs without the packs or
# the private framework. Hermetic twins: the ane engine unit tests, `requests.missingCap`; pack-gated:
# `ANE_TEST_PACK=<pack> zig build test -Dtest-filter="pack on the ANE"`.

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11386}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }
source "$ROOT/tests/_lib_models.sh"
MODEL="${ANE_MODEL:-$(find_model mlx-community/Qwen3.5-0.8B-MLX-4bit lmstudio-community/Qwen3.5-2B-MLX-4bit)}"
OTHER="${ANE_OTHER_MODEL:-$(find_model mlx-community/LFM2.5-2.6B-8bit LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit)}"
[ -n "$MODEL" ] || { echo "SKIP: needs a dense Qwen3.5 pack (ANE_MODEL)"; exit 0; }
[ -d /System/Library/PrivateFrameworks/AppleNeuralEngine.framework ] || { echo "SKIP: no AppleNeuralEngine framework"; exit 0; }

PASS=0; FAIL=0
check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi
}
FAKE_HOME="$(mktemp -d)"
mkdir -p "$FAKE_HOME/.mlx-serve"
export MLX_SERVE_ANE_CACHE_DIR="${MLX_SERVE_ANE_CACHE_DIR:-$HOME/.mlx-serve/ane-cache}"
LOG="$FAKE_HOME/server.log"
SRV=""
stop() {
    [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
    pkill -f "mlx-serve.*--port $PORT" 2>/dev/null
    for _ in $(seq 1 40); do pgrep -f "mlx-serve.*--port $PORT" >/dev/null || break; sleep 0.5; done
    SRV=""
}
cleanup() { stop; rm -rf "$FAKE_HOME"; }
trap cleanup EXIT
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null

boot() { # boot <model> [flags...] — a cold ANE compile takes minutes, a cached load seconds
    local m="$1"; shift
    : >"$LOG"
    HOME="$FAKE_HOME" "$BIN" --serve --model "$m" --host 127.0.0.1 --port "$PORT" --log-level debug --log-file off "$@" >"$LOG" 2>&1 &
    SRV=$!
    srv_wait_health "http://127.0.0.1:$PORT" 1200 0.5 "$SRV" && return 0
    echo "FAIL: server never became healthy"; tail -5 "$LOG"; exit 1
}
meta() { # meta <field> — the default model row's meta.<field>, or a top-level field
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
m = json.load(sys.stdin)['data'][0]
print(m.get(sys.argv[1], m['meta'].get(sys.argv[1])))" "$1"
}
chat() { # chat <json> — prints 'code<TAB>body'
    local out code
    out="$(mktemp)"
    code="$(curl -s -o "$out" -w '%{http_code}' --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$1")"
    printf '%s\t%s' "$code" "$(cat "$out")"
    rm -f "$out"
}
content() { python3 -c "import sys, json; print(json.loads(sys.stdin.read().split('\t', 1)[1])['choices'][0]['message']['content'])"; }
code() { cut -f1 <<<"$1"; }

echo "[1] \"ane\": true in model-settings.json -> served on the Neural Engine ($MODEL)"
printf '{ "%s": { "ane": true } }\n' "$MODEL" >"$FAKE_HOME/.mlx-serve/model-settings.json"
boot "$MODEL"
check "[1] log: [ane] ready" "$(grep -q '\[ane\] ready' "$LOG" && echo 1 || echo 0)"
check "[1] meta.engine ane (got $(meta engine))" "$([ "$(meta engine)" = "ane" ] && echo 1 || echo 0)"
check "[1] context_length 65536 (got $(meta context_length))" "$([ "$(meta context_length)" = "65536" ] && echo 1 || echo 0)"

echo "[2] stream and non-stream deliver the same bytes"
BODY='{"messages":[{"role":"user","content":"Name three primary colors, comma separated."}],"max_tokens":40,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}'
R="$(chat "$BODY}")"
NS="$(content <<<"$R")"
check "[2] non-stream 200 with content" "$([ "$(code "$R")" = "200" ] && [ -n "$NS" ] && echo 1 || echo 0)"
ST="$(curl -sN --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY,\"stream\":true}" | python3 -c "
import sys, json
out = ''
for line in sys.stdin:
    if line.startswith('data: ') and line.strip() != 'data: [DONE]':
        for c in json.loads(line[6:]).get('choices', []):
            out += c.get('delta', {}).get('content') or ''
print(out)")"
check "[2] stream == non-stream" "$([ "$ST" = "$NS" ] && echo 1 || echo 0)"

echo "[3] logprobs are served on the host's logits rows; what the engine cannot serve is refused by name"
R="$(chat "$BODY,\"logprobs\":true,\"top_logprobs\":2}")"
check "[3] logprobs -> 200 with top_logprobs" "$([ "$(code "$R")" = "200" ] && grep -q '"top_logprobs"' <<<"$R" && echo 1 || echo 0)"
R="$(chat '{"messages":[{"role":"user","content":"Call it."}],"max_tokens":20,"tools":[{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}],"tool_choice":"required"}')"
check "[3] forced tool call -> 400 naming the Neural Engine" "$([ "$(code "$R")" = "400" ] && grep -q "Neural Engine" <<<"$R" && echo 1 || echo 0)"
PNG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
R="$(chat "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is this?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$PNG\"}}]}],\"max_tokens\":8}")"
check "[3] image -> 400 naming the Neural Engine" "$([ "$(code "$R")" = "400" ] && grep -q "Neural Engine" <<<"$R" && echo 1 || echo 0)"

echo "[4] a prompt past the first context, then the same prompt from a checkpoint"
LONG="$(python3 -c "
import json
words = ' '.join('alpha%d' % (i % 97) for i in range(5200))
print(json.dumps({'messages': [{'role': 'user', 'content': 'The secret word is BLUEBERRY. ' + words + ' What is the secret word? One word.'}], 'max_tokens': 8, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}))")"
R="$(chat "$LONG")"
check "[4] long prompt -> 200 with the secret word" "$([ "$(code "$R")" = "200" ] && grep -qi "blueberry" <<<"$(content <<<"$R")" && echo 1 || echo 0)"
check "[4] log: context 4096 -> 16384" "$(grep -q 'context 4096 -> 16384' "$LOG" && echo 1 || echo 0)"
R="$(chat "$LONG")"
CACHED="$(python3 -c "import sys, json; print(json.loads(sys.stdin.read().split('\t', 1)[1])['usage']['prompt_tokens_details']['cached_tokens'])" <<<"$R")"
check "[4] repeat: cached_tokens > 0 (got $CACHED)" "$([ "${CACHED:-0}" -gt 0 ] && echo 1 || echo 0)"
check "[4] log: tokens from a checkpoint" "$(grep -q 'from a checkpoint' "$LOG" && echo 1 || echo 0)"
check "[4] no [ane] request failed" "$(grep -q '\[ane\] request failed' "$LOG" && echo 0 || echo 1)"
stop

echo "[5] no opt-in: the same pack stays on the GPU"
rm -f "$FAKE_HOME/.mlx-serve/model-settings.json"
boot "$MODEL"
check "[5] meta.engine not ane (got $(meta engine))" "$([ "$(meta engine)" != "ane" ] && echo 1 || echo 0)"
check "[5] no [ane] lines" "$(grep -q '\[ane\]' "$LOG" && echo 0 || echo 1)"
stop

if [ -n "$OTHER" ]; then
    echo "[6] --ane on a pack the engine cannot run: the GPU serves it, by name ($OTHER)"
    boot "$OTHER" --ane
    check "[6] log names the fallback" "$(grep -q 'does not run on the Neural Engine' "$LOG" && echo 1 || echo 0)"
    check "[6] meta.engine not ane (got $(meta engine))" "$([ "$(meta engine)" != "ane" ] && echo 1 || echo 0)"
    R="$(chat "$BODY}")"
    check "[6] it answers" "$([ "$(code "$R")" = "200" ] && echo 1 || echo 0)"
    stop
else
    echo "  SKIP [6]: no pack the engine cannot run (ANE_OTHER_MODEL)"
fi

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
