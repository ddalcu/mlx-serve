#!/bin/bash
# MTP byte-equivalence test.
#
# Verifies that running the same temp=0 chat completion request against the
# server with --mtp produces *identical* output text to running it without
# --mtp. This is the correctness gate for the MTP draft+verify implementation:
# greedy decode must yield identical token streams whether or not MTP is on.
#
# Requires:
#   - A built mlx-serve binary (run `zig build -Doptimize=ReleaseFast` first)
#   - A model directory with MTP weights present in safetensors
#     (config field `mtp_num_hidden_layers > 0` AND actual `*.mtp.*` tensors)
#
# Most MLX-converted Qwen3.5/Qwen3.6 checkpoints from Hugging Face have the
# config metadata but the conversion strips the MTP weight tensors. Verify
# with: python3 -c "from safetensors import safe_open; \
#   import sys; \
#   [print(k) for k in safe_open(sys.argv[1], framework='pt').keys() if 'mtp' in k.lower()][:5]" \
#   path/to/model.safetensors
#
# Usage:
#   MTP_TEST_MODEL=/path/to/qwen3.5-with-mtp ./tests/test_mtp_equivalence.sh [port]
#
# Without MTP_TEST_MODEL set, the test exits 0 with a skip message — keeps it
# safe for CI matrices that don't have an MTP-bearing checkpoint available.

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if [ -z "$MTP_TEST_MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_mtp_equivalence: \$MTP_TEST_MODEL is not set."
    echo
    echo "  Set MTP_TEST_MODEL to a model directory whose safetensors include"
    echo "  MTP head weights (keys matching '*.mtp.*'). Most MLX-converted"
    echo "  Qwen3.5/3.6 checkpoints have the config field but no weights;"
    echo "  see the comment block in this script for verification commands."
    exit 0
fi

if [ ! -d "$MTP_TEST_MODEL" ]; then
    echo -e "${RED}FAIL${NC} \$MTP_TEST_MODEL=$MTP_TEST_MODEL is not a directory."
    exit 1
fi

if [ ! -f "$MTP_TEST_MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MTP_TEST_MODEL/config.json missing."
    exit 1
fi

# Confirm MTP weights are actually present (not just config metadata).
HAS_MTP_WEIGHTS=$(python3 -c "
import os, sys
from safetensors import safe_open
d = '$MTP_TEST_MODEL'
n = 0
for f in os.listdir(d):
    if f.endswith('.safetensors'):
        try:
            with safe_open(os.path.join(d, f), framework='pt') as st:
                for k in st.keys():
                    if '.mtp.' in k or k.startswith('mtp.'):
                        n += 1
                        break
            if n: break
        except: pass
print('1' if n else '0')
" 2>/dev/null)

if [ "$HAS_MTP_WEIGHTS" != "1" ]; then
    echo -e "${YELLOW}SKIP${NC} test_mtp_equivalence: $MTP_TEST_MODEL config.json may declare MTP layers, but no '*.mtp.*' tensors were found in the safetensors. The MLX conversion likely stripped them."
    exit 0
fi

# Find binary
BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

PROMPT="Write the first line of the Linux kernel boot message."
JSON_PAYLOAD=$(cat <<EOF
{
  "model": "mlx-serve",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": 64,
  "temperature": 0.0,
  "stream": false
}
EOF
)

run_request() {
    # All status messages go to stderr so the captured stdout is JUST the
    # final completion text from the model.
    local label="$1" mtp_flag="$2"
    echo -e "  starting server ($label)..." >&2
    local logfile=$(mktemp)
    "$BINARY" --model "$MTP_TEST_MODEL" --serve --port "$PORT" $mtp_flag > "$logfile" 2>&1 &
    local pid=$!
    # Wait for server up (max 60s — model load can be slow)
    local up=0
    for i in $(seq 1 60); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then
            up=1
            break
        fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        echo -e "  ${RED}FAIL${NC} server did not become healthy in 60s" >&2
        cat "$logfile" | tail -20 >&2
        kill $pid 2>/dev/null || true
        rm -f "$logfile"
        return 1
    fi
    local body
    body=$(echo "$JSON_PAYLOAD" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions")
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    rm -f "$logfile"
    # Extract content
    echo "$body" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

echo "== MTP byte-equivalence test =="
echo "  model: $MTP_TEST_MODEL"
echo "  prompt: $PROMPT"
echo

# Pre-emptively kill any stale server on the test port.
pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

OUT_NOMTP=$(run_request "without --mtp" "--no-mtp") || exit 1
echo "  no-mtp output captured ($(echo "$OUT_NOMTP" | wc -c) bytes)"

sleep 2
OUT_MTP=$(run_request "with --mtp" "--mtp") || exit 1
echo "  with-mtp output captured ($(echo "$OUT_MTP" | wc -c) bytes)"

if [ "$OUT_NOMTP" = "$OUT_MTP" ]; then
    echo -e "${GREEN}PASS${NC} byte-identical output with vs without --mtp"
    exit 0
else
    echo -e "${RED}FAIL${NC} outputs differ:"
    echo "  --no-mtp:"
    echo "$OUT_NOMTP" | sed 's/^/    /'
    echo "  --mtp:"
    echo "$OUT_MTP" | sed 's/^/    /'
    diff <(echo "$OUT_NOMTP") <(echo "$OUT_MTP") | sed 's/^/    /'
    exit 1
fi
