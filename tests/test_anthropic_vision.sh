#!/bin/bash
# Integration test: /v1/messages (Anthropic) accepts and forwards image content
# blocks to the vision encoder.
#
# Bug surfaced by local-llm-bench's vision.solid_red on /v1/messages: the model
# answered "Blue" for a red 48x48 PNG. Same image works on /v1/chat/completions.
# Cause: handleAnthropicMessages only handled `tool_result` and `text` blocks
# in user content arrays — `image` blocks were silently dropped, so the encoder
# never saw the picture and the model guessed.

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

echo "=== /v1/messages vision test ==="
echo "Model: $MODEL_DIR"
echo ""

echo "Starting server..."
./zig-out/bin/mlx-serve \
    --model "$MODEL_DIR" --serve --port $PORT --log-level info \
    >/tmp/mlx-serve-anthropic-vision-test.log 2>&1 &
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

# Generate a 48×48 solid-red PNG as base64. Pure-stdlib (zlib + struct + crc32)
# so no Pillow required.
make_solid_png_b64() {
    local r=$1 g=$2 b=$3
    python3 - "$r" "$g" "$b" <<'PY'
import sys, struct, zlib, base64
r,g,b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
W = H = 48
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t+d) & 0xffffffff)
sig = b"\x89PNG\r\n\x1a\n"
ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)  # 8-bit RGB
raw = b""
for _ in range(H):
    raw += b"\x00" + bytes([r,g,b]) * W
idat = zlib.compress(raw, 9)
png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
print(base64.b64encode(png).decode())
PY
}

ask_anthropic_vision() {
    # $1 = base64 PNG, $2 = question. Returns the assistant text.
    local b64="$1" question="$2"
    local body
    body=$(jq -n --arg b64 "$b64" --arg q "$question" '{
        model:"mlx-serve",
        max_tokens:32,
        temperature:0,
        messages:[
            {role:"user",content:[
                {type:"image",source:{type:"base64",media_type:"image/png",data:$b64}},
                {type:"text",text:$q}
            ]}
        ]
    }')
    curl -sf "$BASE/v1/messages" -H "Content-Type: application/json" \
        -H "anthropic-version: 2023-06-01" -d "$body" \
        | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
parts = r.get("content", [])
print("".join(p["text"] for p in parts if p.get("type")=="text"))
'
}

# ── Test 1: solid red PNG via Anthropic image block ──
echo "--- Test 1: red PNG → /v1/messages ---"
RED_B64=$(make_solid_png_b64 220 30 30)
ANSWER=$(ask_anthropic_vision "$RED_B64" "What color is this image? One word only.")
echo "  reply: $ANSWER"
if echo "$ANSWER" | grep -qiE '\bred\b'; then
    run_test "red PNG recognized" "PASS" ""
else
    run_test "red PNG recognized" "FAIL" "expected /red/, got: $ANSWER"
fi
echo ""

# ── Test 2: solid blue PNG via Anthropic image block ──
echo "--- Test 2: blue PNG → /v1/messages ---"
BLUE_B64=$(make_solid_png_b64 30 60 220)
ANSWER=$(ask_anthropic_vision "$BLUE_B64" "What color is this image? One word only.")
echo "  reply: $ANSWER"
if echo "$ANSWER" | grep -qiE '\bblue\b'; then
    run_test "blue PNG recognized" "PASS" ""
else
    run_test "blue PNG recognized" "FAIL" "expected /blue/, got: $ANSWER"
fi
echo ""

# ── Test 3: image with source.type=url (data URL) — alternate Anthropic shape ──
echo "--- Test 3: red PNG via data-URL source ---"
RED_B64=$(make_solid_png_b64 220 30 30)
DATA_URL="data:image/png;base64,${RED_B64}"
BODY=$(jq -n --arg url "$DATA_URL" '{
    model:"mlx-serve",
    max_tokens:32,
    temperature:0,
    messages:[
        {role:"user",content:[
            {type:"image",source:{type:"url",url:$url}},
            {type:"text",text:"What color is this image? One word only."}
        ]}
    ]
}')
ANSWER=$(curl -sf "$BASE/v1/messages" -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" -d "$BODY" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
parts = r.get("content", [])
print("".join(p["text"] for p in parts if p.get("type")=="text"))
')
echo "  reply: $ANSWER"
if echo "$ANSWER" | grep -qiE '\bred\b'; then
    run_test "red PNG via data-URL recognized" "PASS" ""
else
    run_test "red PNG via data-URL recognized" "FAIL" "expected /red/, got: $ANSWER"
fi
echo ""

echo "=== Result: $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ]
