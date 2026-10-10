#!/bin/bash
# /v1/embeddings end-to-end test (encoder-only BERT models; check 4c: a decoder-arch embedder).
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
#   4c. a decoder-arch embedder (Qwen3-Embedding) matches singles across
#      sub-batches and survives them (skipped when the model is missing)
#   4d. EmbeddingGemma 2: a batch whose long row crosses the sliding band
#      matches its singles with finite vectors, the card's retrieval example
#      ranks Mars first, `dimensions` truncates; `messages` carries images and
#      video (sizes, token counts, order, refusals) (skipped when the model is missing)
#   5. hot-load: a chat-model server embeds via "model": "<encoder-id>"
#      (skipped when the chat model is missing)
#
# Requires:
#   - A built mlx-serve binary (zig build -Doptimize=ReleaseFast)
#   - EMBED_TEST_MODEL or ~/.mlx-serve/models/mlx-community/bge-small-en-v1.5-8bit
#   - (check 4c only) QWEN3_EMBED_TEST_MODEL or ~/.mlx-serve/models/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
#   - (check 4d only) EG2_TEST_MODEL or ~/.mlx-serve/models/google/embeddinggemma-2
#   - (check 5 only) CHAT_TEST_MODEL or ~/.mlx-serve/models/mlx-community/Qwen3-0.6B-nvfp4
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
CHAT_MODEL="${CHAT_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3-0.6B-nvfp4}"
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

# --- 3b. OpenAI `dimensions`: truncate + L2-renormalize, honest 400s ---
# Accepting the parameter and ignoring it silently misleads callers into
# storing wrong-width vectors (llmprobe: "embeddings: dimensions honored").
# Semantics are text-embedding-3's: keep the first N components, renormalize.
python3 - "$BASE" > /tmp/test_embeddings_dims.out <<'EOF'
import json, math, sys, urllib.request, urllib.error

base = sys.argv[1]
def post(body):
    req = urllib.request.Request(base + "/v1/embeddings",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

text = "The quick brown fox jumps over the lazy dog."
_, full = post({"model": "mlx-serve", "input": text})
fv = full["data"][0]["embedding"]

code, r = post({"model": "mlx-serve", "input": text, "dimensions": 64})
v = r["data"][0]["embedding"] if code == 200 else []
norm = math.sqrt(sum(x*x for x in v)) if v else 0.0
# Expected: the full vector's first 64 components, L2-renormalized.
tn = math.sqrt(sum(x*x for x in fv[:64]))
exp = [x / tn for x in fv[:64]]
close = v and max(abs(a-b) for a, b in zip(v, exp)) < 1e-4
print(f"dims64 {1 if code == 200 and len(v) == 64 and abs(norm-1.0) < 1e-3 and close else 0} "
      f"code={code} len={len(v)} norm={norm:.6f}")

code_big, _ = post({"model": "mlx-serve", "input": text, "dimensions": 100000})
print(f"dimsbig {1 if code_big == 400 else 0} code={code_big}")

code_zero, _ = post({"model": "mlx-serve", "input": text, "dimensions": 0})
print(f"dimszero {1 if code_zero == 400 else 0} code={code_zero}")

# `messages` carries images and video: only a multimodal embedder takes it, and says so.
code_msg, body_msg = post({"model": "mlx-serve", "messages": [{"role": "user", "content": text}]})
print(f"messages400 {1 if code_msg == 400 and 'multimodal' in json.dumps(body_msg) else 0} code={code_msg}")
EOF
check "dimensions=64: truncated + L2-renormalized first-64 of the full vector" \
    "$(awk '/^dims64/{print $2}' /tmp/test_embeddings_dims.out)" \
    "$(grep '^dims64' /tmp/test_embeddings_dims.out)"
check "dimensions beyond the model's width returns 400" \
    "$(awk '/^dimsbig/{print $2}' /tmp/test_embeddings_dims.out)" \
    "$(grep '^dimsbig' /tmp/test_embeddings_dims.out)"
check "dimensions=0 returns 400" \
    "$(awk '/^dimszero/{print $2}' /tmp/test_embeddings_dims.out)" \
    "$(grep '^dimszero' /tmp/test_embeddings_dims.out)"
check "messages on a text-only encoder returns 400 naming the multimodal models" \
    "$(awk '/^messages400/{print $2}' /tmp/test_embeddings_dims.out)" \
    "$(grep '^messages400' /tmp/test_embeddings_dims.out)"

# --- 4. generation rejected on encoder-only ---
GEN_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 30 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"mlx-serve","messages":[{"role":"user","content":"hi"}]}')
check "chat completion on encoder-only model returns 400" \
    "$([ "$GEN_CODE" = "400" ] && echo 1 || echo 0)" "got HTTP $GEN_CODE"

# --- 4b. pooling signal + embedding input ceiling (issues #116/#117) ---
# bge-small is a CLS-pooling BERT (its card sets pooling_mode_cls_token); the
# mlx-community conversion ships no pooling metadata, so the known-family name
# fallback must engage — the boot log is the observable.
if [ "$(basename "$EMBED_MODEL")" = "bge-small-en-v1.5-8bit" ]; then
    check "CLS pooling inferred from checkpoint name at load (issue #116)" \
        "$(grep -q 'pooling inferred from checkpoint name: cls' /tmp/test_embeddings_server.log && echo 1 || echo 0)"
fi

# Ready model surfaces the effective embedding input ceiling (issue #117):
# with no --embedding-max-length flag, auto = the model's declared window.
META_LIMIT=$(curl -s -m 30 "$BASE/v1/models" | python3 -c "
import json, sys
for m in json.load(sys.stdin)['data']:
    meta = m.get('meta') or {}
    if m.get('state') == 'ready' and meta.get('embedding_max_length') is not None:
        print(meta['embedding_max_length']); break
")
check "ready encoder advertises meta.embedding_max_length (auto = model window)" \
    "$([ -n "$META_LIMIT" ] && [ "$META_LIMIT" -gt 0 ] && echo 1 || echo 0)" "got '$META_LIMIT'"

# Over-window input: an explicit structured 400 naming the input index and
# both counts — never a silent truncation (issue #117). A word is at least one
# token, so window+100 words is past the model's window; the index must
# identify the SECOND input.
LONG_INPUT=$(python3 -c "print(' '.join(['tokenized']*($META_LIMIT + 100)))")
OVER_RESP=$(embed "[\"short one\", \"$LONG_INPUT\"]")
check "over-limit input earns a 400 naming index + counts (issue #117)" \
    "$(echo "$OVER_RESP" | grep -q 'Input at index 1 exceeds the maximum embedding input length' && echo 1 || echo 0)" \
    "$(echo "$OVER_RESP" | head -c 200)"

stop_server

# --- 4c. decoder-arch embedder across sub-batches ---
# Qwen3-Embedding forwards through the KV cache, and a request past EMBED_TOKEN_BUDGET
# (64 x 512 padded tokens) splits into sub-batches that must each start from an empty one.
QWEN3_EMBED_MODEL="${QWEN3_EMBED_TEST_MODEL:-$HOME/.mlx-serve/models/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ}"
echo "=== /v1/embeddings: decoder-arch embedder across sub-batches ==="
if [ ! -d "$QWEN3_EMBED_MODEL" ]; then
    echo -e "  ${YELLOW}SKIP${NC} Qwen3-Embedding model not found at $QWEN3_EMBED_MODEL"
else
    start_server /tmp/test_embeddings_qwen3.log --model "$QWEN3_EMBED_MODEL" --log-level info
    python3 - "$BASE" > /tmp/test_embeddings_qwen3.out <<'EOF'
import json, math, random, sys, urllib.request

base = sys.argv[1]
rng = random.Random(7)
syl = ["ka", "lo", "mi", "ren", "tu", "vas", "po", "shi", "den", "gal", "or", "fe"]
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))
def embed(inputs):
    rows = sorted(post("/v1/embeddings", {"model": "mlx-serve", "input": inputs})["data"],
                  key=lambda d: d["index"])
    return [d["embedding"] for d in rows]
def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
# The embed path appends one token (EOS) to what /tokenize reports.
ntok = lambda t: len(post("/tokenize", {"content": t})["tokens"]) + 1
word = lambda: "".join(rng.choice(syl) for _ in range(rng.randint(1, 3)))
def text(lo, hi):  # distinct pseudo-words until the token count lands in [lo, hi]
    ws = [word()]
    for _ in range(500):
        n = ntok(" ".join(ws))
        if lo <= n <= hi: return " ".join(ws)
        if n < lo: ws += [word() for _ in range(max(1, (lo - n) // 4))]
        else: ws.pop()
    raise RuntimeError(f"no text of {lo}-{hi} tokens")

# Sizes straddle EMBED_TOKEN_BUDGET: rows x padded length decides where a request splits.
BUDGET = 64 * 512
short = [text(8, 12) for _ in range(35)]

# Equal sub-batches: 32 short, then 32 long that fit only 32 to a sub-batch. The later
# sub-batch must match the same 32 inputs sent as their own request.
try:
    long_ = [text(BUDGET // 33 + 1, BUDGET // 32) for _ in range(32)]
    scores = [cos(a, b) for a, b in zip(embed(short[:32] + long_)[32:], embed(long_))]
    worst = min(scores) if all(map(math.isfinite, scores)) else float("nan")
    print(f"equal {1 if worst >= 0.999 else 0} worst-cosine={worst:.5f}")
except Exception as e:
    print(f"equal 0 {e}")

# A second sub-batch with more rows, on the Ollama route: 29 long that fit only 29, then 35 short.
try:
    inputs = [text(BUDGET // 30 + 1, BUDGET // 29) for _ in range(29)] + short
    rows = post("/api/embed", {"model": "mlx-serve", "input": inputs})["embeddings"]
    ok = len(rows) == 64 and all(cos(rows[i], embed([inputs[i]])[0]) >= 0.995 for i in (0, 28, 29, 63))
    print(f"larger {1 if ok else 0} rows={len(rows)}")
except Exception as e:
    print(f"larger 0 {e}")
try:
    alive = len(embed(["still serving"])) == 1
except Exception as e:
    alive = False
print(f"alive {1 if alive else 0}")
EOF
    check "equal sub-batches: the later one matches the same inputs sent alone (cosine >= 0.999)" \
        "$(awk '/^equal/{print $2}' /tmp/test_embeddings_qwen3.out)" \
        "$(grep '^equal' /tmp/test_embeddings_qwen3.out)"
    check "second sub-batch with more rows (/api/embed) answers 200, sampled rows match singles" \
        "$(awk '/^larger/{print $2}' /tmp/test_embeddings_qwen3.out)" \
        "$(grep '^larger' /tmp/test_embeddings_qwen3.out)"
    check "server still answers after the multi-sub-batch requests" \
        "$(awk '/^alive/{print $2}' /tmp/test_embeddings_qwen3.out)" \
        "$(tail -3 /tmp/test_embeddings_qwen3.log)"
    stop_server
fi

# --- 4d. EmbeddingGemma 2 (bidirectional Gemma 4 trunk, sliding band + full layers) ---
# A batch pads its short rows out to the long one. Past the band radius (512) a padded query sees
# only padding in the sliding layers, and a NaN row there reaches the real rows through V in every
# later layer: the long row below crosses the band, and every vector must come back finite.
EG2_MODEL="${EG2_TEST_MODEL:-$HOME/.mlx-serve/models/google/embeddinggemma-2}"
echo "=== /v1/embeddings: EmbeddingGemma 2 ==="
if [ ! -d "$EG2_MODEL" ]; then
    echo -e "  ${YELLOW}SKIP${NC} EmbeddingGemma 2 not found at $EG2_MODEL"
else
    start_server /tmp/test_embeddings_eg2.log --model "$EG2_MODEL" --log-level info
    python3 - "$BASE" > /tmp/test_embeddings_eg2.out <<'EOF'
import json, math, sys, urllib.request, urllib.error

base = sys.argv[1]
def call(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
def embed(inputs, **kw):
    code, r = call("/v1/embeddings", {"model": "mlx-serve", "input": inputs, **kw})
    return code, ([d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])] if code == 200 else r)
def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
finite = lambda v: all(math.isfinite(x) for x in v)

_, models = call("/v1/models")
m = next(m for m in models["data"] if m.get("state") == "ready")
ok = (m["meta"]["architecture"] == "embedding_gemma2" and "embeddings" in m["capabilities"]
      and m["context_length"] == 8192 and m["meta"]["embedding_max_length"] == 8192
      and m["input_modalities"] == ["text", "image", "video"])
print(f"meta {1 if ok else 0} {m['meta']['architecture']} ctx={m['context_length']} modalities={m['input_modalities']}")

short = ["task: search result | query: Which planet is known as the Red Planet?",
         "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
         "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet."]
long_text = " ".join(f"item{i} {w}" for i, w in zip(range(400), ["ka", "lo", "mi", "ren", "tu", "vas"] * 80))
_, ntok = call("/tokenize", {"content": long_text})
code, vs = embed(short)
unit = code == 200 and all(len(v) == 768 and abs(math.sqrt(sum(x*x for x in v)) - 1) < 1e-3 for v in vs)
print(f"shape {1 if unit else 0} code={code}")

# The model card's own retrieval example: the query is closer to Mars than to Venus.
print(f"ranking {1 if unit and cos(vs[0], vs[2]) > cos(vs[0], vs[1]) else 0}")

# Mixed batch: the long row crosses the radius-512 band and pads the short rows past it.
code, batch = embed(short + [long_text])
singles = [embed([t])[1][0] for t in short + [long_text]] if code == 200 else []
worst = min((cos(a, b) for a, b in zip(batch, singles)), default=float("nan")) if code == 200 else float("nan")
good = code == 200 and all(finite(v) for v in batch) and worst >= 0.999 and len(ntok["tokens"]) > 600
print(f"batch {1 if good else 0} worst-cosine={worst:.5f} long-tokens={len(ntok['tokens'])}")

# Matryoshka: the first N components, renormalized.
code, d256 = embed(short[:1], dimensions=256)
tn = math.sqrt(sum(x*x for x in vs[0][:256]))
mrl = code == 200 and len(d256[0]) == 256 and max(abs(a - b / tn) for a, b in zip(d256[0], vs[0][:256])) < 1e-4
print(f"dims256 {1 if mrl else 0}")

# Past the trained window: a structured 400 naming the input, never a truncation.
code, r = embed(["short one", " ".join(["tokenized"] * 9000)])
msg = json.dumps(r)
print(f"overlimit {1 if code == 400 and 'index 1' in msg and '8192' in msg else 0} code={code}")
code, _ = call("/v1/chat/completions", {"model": "mlx-serve", "messages": [{"role": "user", "content": "hi"}]})
print(f"chat400 {1 if code == 400 else 0} code={code}")

# ── images and video through `messages` ──
import base64, struct, zlib
def png(w, h, px):
    raw = b"".join(b"\0" + bytes(c for x in range(w) for c in px(x, y)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")).decode()
sky = png(160, 96, lambda x, y: (30 + y, 90 + y // 2, 200 - x // 2))
brick = png(120, 120, lambda x, y: (180 + (x * y) % 40, 60 + (x // 8 % 2) * 50, 40))
def soft_tokens(h, w, budget):  # the processor's target size (sides in multiples of 48 that fit the budget), as pooled patches
    side = 48
    f = math.sqrt(budget * 9 * 256 / (h * w))
    return int(f * h // side) * int(f * w // side)
text = lambda s: {"type": "text", "text": s}
image = lambda url: {"type": "image_url", "image_url": {"url": url}}
video = lambda frames: {"type": "video_url", "video_url": {"frames": frames}}
def mm(*parts, **kw):
    return call("/v1/embeddings", {"model": "mlx-serve", **kw, "messages": [{"role": "user", "content": list(parts)}]})
vec = lambda r: r[1]["data"][0]["embedding"]
used = lambda r: r[1]["usage"]["prompt_tokens"] if r[0] == 200 else r[1]

res = mm(image(sky))
v_sky = vec(res) if res[0] == 200 else []
want = 2 + 2 + soft_tokens(96, 160, 280)
print(f"mm_shape {1 if res[0] == 200 and len(v_sky) == 768 and abs(math.sqrt(sum(x*x for x in v_sky)) - 1) < 1e-3 and used(res) == want else 0} code={res[0]} tokens={used(res)} want={want}")
v_again, v_brick = vec(mm(image(sky))), vec(mm(image(brick)))
print(f"mm_stable {1 if v_again == v_sky and cos(v_sky, v_brick) < 0.99 else 0} different-image-cosine={cos(v_sky, v_brick):.4f}")

# Order is part of the input: text before the image is not text after it.
before, after = vec(mm(text("a picture of "), image(sky))), vec(mm(image(sky), text(" a picture of")))
print(f"mm_order {1 if cos(before, after) < 0.9999 and cos(before, v_sky) < 0.9999 else 0} cosine={cos(before, after):.5f}")

# Plain text through `messages` is the text path (same <bos> … <eos> wrap, same vector).
t_msg, t_in = vec(mm(text("Mars, known for its reddish appearance"))), embed(["Mars, known for its reddish appearance"])[1][0]
print(f"mm_text {1 if cos(t_msg, t_in) > 0.99999 else 0} cosine={cos(t_msg, t_in):.6f}")

# Video: a wrapped block per frame at the frame budget; more frames than the cap are cut to the cap.
frame_tokens = 2 + soft_tokens(64, 96, 140)
clip3 = mm(video([png(96, 64, lambda x, y: (x * 2, y * 3, 128))] * 3))
clip40 = mm(video([png(96, 64, lambda x, y, i=i: (x * 2, y * 3, 3 * i)) for i in range(40)]))
print(f"mm_video {1 if used(clip3) == 2 + 3 * frame_tokens and used(clip40) == 2 + 32 * frame_tokens else 0} tokens={used(clip3)}/{2 + 3 * frame_tokens} {used(clip40)}/{2 + 32 * frame_tokens}")

# Refusals are named, and nothing is half-served.
def refused(r, needle):
    return r[0] == 400 and needle in json.dumps(r[1])
audio = {"type": "input_audio", "input_audio": {"data": "AAAA"}}
bad = [
    refused(call("/v1/embeddings", {"model": "mlx-serve", "input": "x", "messages": [{"role": "user", "content": "y"}]}), "not both"),
    refused(mm(image("data:image/png;base64,AAAA")), "could not be decoded"),
    refused(mm(audio), "audio"),
    refused(call("/v1/embeddings", {"model": "mlx-serve", "messages": []}), "non-empty"),
    refused(mm(video([])), "frames"),
]
print(f"mm_refused {1 if all(bad) else 0} {bad}")

# Past the window: 40 images are ~10k tokens; refused up front, before any of them reaches the tower.
over = mm(*[image(sky)] * 40)
print(f"mm_limit {1 if refused(over, 'index 0') and '8192' in json.dumps(over[1]) else 0} code={over[0]}")
EOF
    check "EmbeddingGemma 2 advertises embeddings, an 8192 window and text, image and video input" \
        "$(awk '/^meta/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^meta' /tmp/test_embeddings_eg2.out)"
    check "768-wide unit vectors for a retrieval batch" \
        "$(awk '/^shape/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^shape' /tmp/test_embeddings_eg2.out)"
    check "the card's retrieval example ranks Mars above Venus" \
        "$(awk '/^ranking/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^ranking' /tmp/test_embeddings_eg2.out)"
    check "a batch with a row past the sliding band: finite, and every row matches its single (cosine >= 0.999)" \
        "$(awk '/^batch/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^batch' /tmp/test_embeddings_eg2.out)"
    check "dimensions=256 keeps the first 256 components, renormalized" \
        "$(awk '/^dims256/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^dims256' /tmp/test_embeddings_eg2.out)"
    check "an input past the 8192 window earns a 400 naming its index" \
        "$(awk '/^overlimit/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^overlimit' /tmp/test_embeddings_eg2.out)"
    check "chat completion on EmbeddingGemma 2 returns 400" \
        "$(awk '/^chat400/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^chat400' /tmp/test_embeddings_eg2.out)"
    check "an image through messages: a 768-wide unit vector, <bos><boi>…<eoi><eos> tokens at the processor's size" \
        "$(awk '/^mm_shape/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^mm_shape' /tmp/test_embeddings_eg2.out)"
    check "the same image embeds identically twice; a different image does not" \
        "$(awk '/^mm_stable/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^mm_stable' /tmp/test_embeddings_eg2.out)"
    check "text before an image and text after it are different inputs" \
        "$(awk '/^mm_order/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^mm_order' /tmp/test_embeddings_eg2.out)"
    check "text-only messages embed exactly like the same text as input" \
        "$(awk '/^mm_text/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^mm_text' /tmp/test_embeddings_eg2.out)"
    check "video: one wrapped block per frame; 40 frames are cut to the 32-frame cap" \
        "$(awk '/^mm_video/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^mm_video' /tmp/test_embeddings_eg2.out)"
    check "input+messages, a bad image, audio, empty messages and empty frames are refused by name" \
        "$(awk '/^mm_refused/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^mm_refused' /tmp/test_embeddings_eg2.out)"
    check "40 images past the window earn a 400 naming the length before the tower runs" \
        "$(awk '/^mm_limit/{print $2}' /tmp/test_embeddings_eg2.out)" "$(grep '^mm_limit' /tmp/test_embeddings_eg2.out)"
    stop_server
fi

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
