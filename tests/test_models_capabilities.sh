#!/bin/bash
# Integration test: /v1/models advertises capabilities, input_modalities, and
# uses the loaded model's full name (directory basename) as the `id`.
#
# Currently /v1/models returns architecture metadata only (vocab/layers/etc)
# and uses the bare architecture family (e.g. "gemma4") as the id. Clients
# have no way to tell which specific quantization/variant is loaded, or
# whether the model supports tools/vision/etc.
#
# Conventions followed:
#   • id        → directory basename (LM Studio / HF style: "gemma-4-e4b-it-8bit")
#   • capabilities[] → Ollama / LM Studio (chat, tool_use, vision, ...)
#   • input_modalities[] → Anthropic (text, image)
#   • meta.architecture → architecture family string (was id before)

set -u

MODEL_DIR=${1:-~/.mlx-serve/models/mlx-community/gemma-4-e4b-it-8bit}
PORT=${2:-8099}
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0
TOTAL=0

if [ ! -d "$MODEL_DIR" ]; then
    echo "SKIP: Model not found at $MODEL_DIR"
    exit 0
fi

if [ ! -x "./zig-out/bin/mlx-serve" ]; then
    echo "FAIL: mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

echo "=== /v1/models capabilities test ==="
echo "Model: $MODEL_DIR"
echo ""

echo "Starting server..."
./zig-out/bin/mlx-serve \
    --model "$MODEL_DIR" --serve --port $PORT --log-level info \
    >/tmp/mlx-serve-models-test.log 2>&1 &
SERVER_PID=$!
sleep 2

for i in $(seq 1 30); do
    if curl -sf "$BASE/health" > /dev/null 2>&1; then
        echo "Server ready (PID $SERVER_PID)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "FAIL: Server did not start within 30s"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done
echo ""

cleanup() {
    echo ""
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

run_test() {
    local name="$1"
    local result="$2"
    local detail="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $name"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name — $detail"
    fi
}

EXPECTED_ID=$(basename "$MODEL_DIR")
RESULT=$(curl -sf "$BASE/v1/models")
echo "raw: $RESULT"
echo ""

# ── Test 1: response shape unchanged (object, data[]) ──
SHAPE=$(echo "$RESULT" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
assert r.get("object") == "list", "object=" + repr(r.get("object"))
assert isinstance(r.get("data"), list) and len(r["data"]) >= 1, "data[] missing or empty"
m = r["data"][0]
for k in ("id","object","created","owned_by"):
    assert k in m, "missing " + k
print("ok")
' 2>&1)
run_test "shape unchanged: list/data/id" "$( [ "$SHAPE" = ok ] && echo PASS || echo FAIL )" "$SHAPE"

# ── Test 1b: id is the full model directory basename ──
ID_OK=$(echo "$RESULT" | python3 -c "
import sys, json
r = json.loads(sys.stdin.read())
got = r['data'][0]['id']
expected = '$EXPECTED_ID'
print('ok' if got == expected else 'fail:got=' + repr(got) + ',want=' + repr(expected))
" 2>&1)
run_test "id is full model name (basename)" \
    "$( [ "$ID_OK" = ok ] && echo PASS || echo FAIL )" "$ID_OK"

# ── Test 1c: meta.architecture exposes architecture family ──
ARCH_OK=$(echo "$RESULT" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
arch = r["data"][0].get("meta",{}).get("architecture")
assert isinstance(arch, str) and len(arch) > 0, "meta.architecture missing or empty: " + repr(arch)
print("ok:" + arch)
' 2>&1)
run_test "meta.architecture present" \
    "$( [[ "$ARCH_OK" == ok:* ]] && echo PASS || echo FAIL )" "$ARCH_OK"

# ── Test 2: capabilities array present and includes core capabilities ──
CAPS=$(echo "$RESULT" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
m = r["data"][0]
caps = m.get("capabilities")
assert isinstance(caps, list), f"capabilities missing or not list: {caps!r}"
# Every chat-able model must advertise at least these:
for required in ("chat","tool_use","streaming"):
    assert required in caps, f"missing capability {required!r} in {caps!r}"
print("ok:" + ",".join(caps))
' 2>&1)
run_test "capabilities[] includes chat/tool_use/streaming" \
    "$( [[ "$CAPS" == ok:* ]] && echo PASS || echo FAIL )" "$CAPS"

# ── Test 3: input_modalities array present and includes text ──
MODS=$(echo "$RESULT" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
m = r["data"][0]
mods = m.get("input_modalities")
assert isinstance(mods, list), f"input_modalities missing or not list: {mods!r}"
assert "text" in mods, f"text not in input_modalities {mods!r}"
print("ok:" + ",".join(mods))
' 2>&1)
run_test "input_modalities[] includes text" \
    "$( [[ "$MODS" == ok:* ]] && echo PASS || echo FAIL )" "$MODS"

# ── Test 4: vision model advertises vision + image ──
# Gemma 4 has a vision encoder; if it loaded, both should appear.
VIS=$(echo "$RESULT" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
m = r["data"][0]
caps = m.get("capabilities") or []
mods = m.get("input_modalities") or []
mid = m.get("id","")
# This test only asserts when a vision-capable model is loaded.
if "gemma4" not in mid and "gemma-4" not in mid:
    print("skip:not_vision_model")
    sys.exit()
if "vision" not in caps:
    print(f"fail:vision_missing_from_caps:{caps}")
    sys.exit()
if "image" not in mods:
    print(f"fail:image_missing_from_modalities:{mods}")
    sys.exit()
print("ok")
' 2>&1)
case "$VIS" in
    ok) run_test "vision model advertises vision+image" "PASS" "" ;;
    skip:*) echo "  SKIP: vision case (non-vision model)" ;;
    *) run_test "vision model advertises vision+image" "FAIL" "$VIS" ;;
esac

# ── Test 4b: context advertised at the ROW top level, not just meta ──
# openai-models-list discovery clients (oh-my-pi, vLLM-shaped readers) look for
# top-level `max_model_len` / `context_length` on each /v1/models row and fall
# back to their own defaults (omp: 128k) when absent — issue #188. Both must
# mirror meta.context_length.
TOPCTX=$(echo "$RESULT" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
for m in r["data"]:
    meta_ctx = (m.get("meta") or {}).get("context_length")
    if meta_ctx is None:
        continue  # rows with no readable config advertise nothing
    mid = m.get("id")
    top = m.get("context_length")
    mml = m.get("max_model_len")
    if top != meta_ctx:
        print("fail:top_level_context_length:%s:%r!=%r" % (mid, top, meta_ctx))
        sys.exit()
    if mml != meta_ctx:
        print("fail:top_level_max_model_len:%s:%r!=%r" % (mid, mml, meta_ctx))
        sys.exit()
print("ok")
' 2>&1)
run_test "top-level context_length/max_model_len mirror meta" \
    "$( [ "$TOPCTX" = ok ] && echo PASS || echo FAIL )" "$TOPCTX"

# ── Test 5: existing meta block still present (no regression) ──
META=$(echo "$RESULT" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
m = r["data"][0]
meta = m.get("meta")
assert isinstance(meta, dict), f"meta missing: {meta!r}"
for k in ("vocab_size","hidden_size","num_layers","quantization","context_length"):
    assert k in meta, f"meta.{k} missing"
print("ok")
' 2>&1)
run_test "existing meta block preserved" \
    "$( [ "$META" = ok ] && echo PASS || echo FAIL )" "$META"

echo ""
echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
