#!/bin/bash
# /v1/embeddings end-to-end test (encoder-only BERT models).
#
# Embeddings are served by a batched GPU forward: every request's inputs are
# tokenized up front, padded per EMBED_MAX_BATCH chunk, and run through ONE
# masked encoder forward instead of one forward per text. The padding mask +
# masked mean-pool are exactly what a per-text loop computes, so batched
# results must match singles — this script pins that equivalence from the
# HTTP surface, where a mask bug would silently skew every padded row.
#
# This test asserts:
#   1. single input: OpenAI list shape, non-empty unit-norm vector, usage
#   2. mixed-length batch: each row cosine-matches its single-input result
#      (>= 0.999), distinct texts stay distinct, order preserved
#   3. a batch larger than EMBED_MAX_BATCH (80 inputs) round-trips intact
#   4. generation endpoints reject encoder-only models with a 400
#   5. hot-load: a chat-model server embeds via "model": "<encoder-id>"
#      (skipped when the chat model is missing)
#
# Requires:
#   - A built mlx-serve binary (zig build -Doptimize=ReleaseFast)
#   - EMBED_TEST_MODEL or ~/.mlx-serve/models/mlx-community/bge-small-en-v1.5-8bit
#   - (check 5 only) CHAT_TEST_MODEL or ~/.mlx-serve/models/Qwen3-0.6B-nvfp4
#
# Usage: ./tests/test_embeddings.sh [port]

set -e

PORT=${1:-11329}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

EMBED_MODEL="${EMBED_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/bge-small-en-v1.5-8bit}"
CHAT_MODEL="${CHAT_TEST_MODEL:-$HOME/.mlx-serve/models/Qwen3-0.6B-nvfp4}"
if [ ! -d "$EMBED_MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_embeddings: encoder model not found at $EMBED_MODEL"
    exit 0
fi
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found. Build with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

FAILURES=0
check() {
    local desc="$1" ok="$2" detail="$3"
    if [ "$ok" = "1" ]; then
        echo -e "  ${GREEN}PASS${NC} $desc"
    else
        echo -e "  ${RED}FAIL${NC} $desc"
        [ -n "$detail" ] && echo "    $detail"
        FAILURES=$((FAILURES + 1))
    fi
}

start_server() {
    local logfile="$1"; shift
    "$BINARY" --serve --port "$PORT" "$@" > "$logfile" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 60); do
        curl -s -f "$BASE/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    echo -e "${RED}FAIL${NC} server did not become healthy"; tail -5 "$logfile"; return 1
}
stop_server() { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; }
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

embed() { # embed <json-input> [model]
    local input="$1" model="${2:-mlx-serve}"
    curl -s -m 120 "$BASE/v1/embeddings" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"input\":$input}"
}

echo "=== /v1/embeddings: encoder-only default model ==="
start_server /tmp/test_embeddings_server.log --model "$EMBED_MODEL" --log-level info

# --- 1. single input shape + unit norm ---
SINGLE_OK=$(embed '"The quick brown fox jumps over the lazy dog."' | python3 -c "
import sys, json, math
r = json.load(sys.stdin)
d = r['data']
v = d[0]['embedding']
ok = (r['object'] == 'list' and len(d) == 1 and d[0]['index'] == 0
      and len(v) > 0 and abs(math.sqrt(sum(x*x for x in v)) - 1.0) < 1e-3
      and r['usage']['prompt_tokens'] > 0)
print(1 if ok else 0)")
check "single input: list shape, unit-norm vector, usage tokens" "$SINGLE_OK"

# --- 2 + 3. batched == singles, order preserved, > EMBED_MAX_BATCH ---
python3 - "$BASE" > /tmp/test_embeddings_batch.out <<'EOF'
import json, math, sys, urllib.request

base = sys.argv[1]
def post(inputs):
    req = urllib.request.Request(base + "/v1/embeddings",
        data=json.dumps({"model": "mlx-serve", "input": inputs}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    rows = sorted(r["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in rows]

def cos(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

# Deliberately mixed lengths so the padded batch exercises the key mask.
texts = [
    "short",
    "Customer frustration was HIGH after the June billing update. " * 8,
    "The quick brown fox jumps over the lazy dog.",
    "Zig is a general-purpose programming language and toolchain. " * 5,
]
singles = [post([t])[0] for t in texts]
batch = post(texts)
worst = min(cos(s, b) for s, b in zip(singles, batch))
print(f"parity {1 if worst >= 0.999 else 0} worst-cosine={worst:.6f}")
distinct = cos(batch[0], batch[1])
print(f"distinct {1 if distinct < 0.99 else 0} cross-cosine={distinct:.4f}")

# 80 inputs > EMBED_MAX_BATCH (64): all rows return, in order, matching singles.
big = [f"document number {i} about topic {i % 7}" for i in range(80)]
big_rows = post(big)
spot = [0, 63, 64, 79]  # both sides of the chunk boundary
spot_ok = all(cos(post([big[i]])[0], big_rows[i]) >= 0.999 for i in spot)
print(f"bigbatch {1 if len(big_rows) == 80 and spot_ok else 0} rows={len(big_rows)}")
EOF
check "mixed-length batch matches single-input results (cosine >= 0.999)" \
    "$(awk '/^parity/{print $2}' /tmp/test_embeddings_batch.out)" \
    "$(grep '^parity' /tmp/test_embeddings_batch.out)"
check "distinct texts stay distinct under batching" \
    "$(awk '/^distinct/{print $2}' /tmp/test_embeddings_batch.out)" \
    "$(grep '^distinct' /tmp/test_embeddings_batch.out)"
check "80-input batch (> EMBED_MAX_BATCH) returns all rows in order" \
    "$(awk '/^bigbatch/{print $2}' /tmp/test_embeddings_batch.out)" \
    "$(grep '^bigbatch' /tmp/test_embeddings_batch.out)"

# --- 4. generation rejected on encoder-only ---
GEN_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 30 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"hi"}]}')
check "chat completion on encoder-only model returns 400" \
    "$([ "$GEN_CODE" = "400" ] && echo 1 || echo 0)" "got HTTP $GEN_CODE"

stop_server

# --- 5. hot-load encoder alongside a chat default ---
echo "=== /v1/embeddings: hot-load alongside chat model ==="
if [ ! -d "$CHAT_MODEL" ]; then
    echo -e "  ${YELLOW}SKIP${NC} chat model not found at $CHAT_MODEL"
else
    ENCODER_ID=$(basename "$EMBED_MODEL")
    start_server /tmp/test_embeddings_hotload.log \
        --model "$CHAT_MODEL" --model-dir "$(dirname "$EMBED_MODEL")" --log-level info
    # The encoder is still an UNLOADED stub here — clients (the app's
    # document indexer) must be able to spot it by capability without
    # paying for a cold load.
    STUB_CAP_OK=$(curl -s -m 30 "$BASE/v1/models" | python3 -c "
import sys, json
r = json.load(sys.stdin)
e = next((m for m in r['data'] if m['id'] == '$ENCODER_ID'), None)
print(1 if e is not None and not e.get('loaded', True)
      and 'embeddings' in e.get('capabilities', []) else 0)")
    check "unloaded encoder stub advertises embeddings capability" "$STUB_CAP_OK"
    HOT_OK=$(embed '"hello world"' "$ENCODER_ID" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    print(1 if len(r['data'][0]['embedding']) > 0 else 0)
except Exception:
    print(0)")
    check "encoder hot-loads by id next to chat default" "$HOT_OK" \
        "$(tail -3 /tmp/test_embeddings_hotload.log)"
    stop_server

    # --- 6. load-by-path: encoder OUTSIDE the server's --model-dir scope ---
    # The app auto-downloads the encoder and registers it via
    # POST /v1/load-model {"model": "<abs path>"} — this must work even when
    # discovery never saw the directory.
    echo "=== /v1/load-model: register encoder by absolute path ==="
    start_server /tmp/test_embeddings_bypath.log \
        --model "$CHAT_MODEL" --model-dir "$(dirname "$CHAT_MODEL")" --log-level info
    NOT_LISTED=$(curl -s -m 30 "$BASE/v1/models" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(0 if any(m['id'] == '$ENCODER_ID' for m in r['data']) else 1)")
    if [ "$NOT_LISTED" != "1" ]; then
        echo -e "  ${YELLOW}SKIP${NC} encoder unexpectedly inside --model-dir scope; can't exercise load-by-path"
    else
        # JSON-escaped slashes (\/) — the exact bytes Swift's JSONSerialization
        # emits. A scanner that doesn't unescape sees "\/Users\/…" and misses
        # the absolute-path branch (live failure 2026-06-12).
        ESCAPED_MODEL=$(printf '%s' "$EMBED_MODEL" | sed 's|/|\\/|g')
        LOAD_CODE=$(curl -s -o /tmp/test_embeddings_load.out -w "%{http_code}" -m 180 \
            "$BASE/v1/load-model" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$ESCAPED_MODEL\"}")
        check "load-model accepts an absolute encoder path (JSON-escaped slashes)" \
            "$([ "$LOAD_CODE" = "200" ] && echo 1 || echo 0)" \
            "HTTP $LOAD_CODE: $(cat /tmp/test_embeddings_load.out)"
        LOAD_CODE2=$(curl -s -o /dev/null -w "%{http_code}" -m 180 \
            "$BASE/v1/load-model" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$EMBED_MODEL\"}")
        check "load-model accepts an absolute encoder path (plain)" \
            "$([ "$LOAD_CODE2" = "200" ] && echo 1 || echo 0)" "HTTP $LOAD_CODE2"
        # The id must be REGISTERED now (embeddings alone can pass spuriously
        # via unknown-id fallback to the default chat model).
        LISTED_NOW=$(curl -s -m 30 "$BASE/v1/models" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(1 if any(m['id'] == '$ENCODER_ID' for m in r['data']) else 0)")
        check "path-registered encoder appears in /v1/models" "$LISTED_NOW"
        BYPATH_OK=$(embed '"hello world"' "$ENCODER_ID" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    print(1 if len(r['data'][0]['embedding']) > 0 else 0)
except Exception:
    print(0)")
        check "path-registered encoder serves embeddings by id" "$BYPATH_OK" \
            "$(tail -3 /tmp/test_embeddings_bypath.log)"
        # Garbage paths must not register anything.
        BAD_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 30 "$BASE/v1/load-model" \
            -H 'Content-Type: application/json' -d '{"model":"/nonexistent/model-dir"}')
        check "bogus absolute path is rejected with 404" \
            "$([ "$BAD_CODE" = "404" ] && echo 1 || echo 0)" "got HTTP $BAD_CODE"
    fi
    stop_server
fi

echo
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}test_embeddings: all checks passed${NC}"
else
    echo -e "${RED}test_embeddings: $FAILURES check(s) failed${NC}"
    exit 1
fi
