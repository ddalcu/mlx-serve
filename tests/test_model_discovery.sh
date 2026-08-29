#!/bin/bash
# test_model_discovery.sh — plan 05 Phase 1 (discovery + listing).
#
# Validates:
#   1. --model-dir flag scans the directory for subdirectories with config.json
#   2. /v1/models lists ALL discovered models (loaded + siblings)
#   3. Loaded model has loaded:true; siblings have loaded:false
#   4. Sibling entries include bytes_on_disk where available
#   5. --model-dir without --model auto-selects the first discovered as loaded
#
# Acceptance: all five assertions pass.
#
# Usage: ./tests/test_model_discovery.sh [models_root] [port]

set -uo pipefail

MODELS_ROOT="${1:-$HOME/.mlx-serve/models}"
PORT="${2:-19060}"
LOADED_MODEL="${3:-gemma-4-e4b-it-4bit}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"

[[ -x "$BINARY" ]] || { echo "Build first" >&2; exit 1; }
[[ -d "$MODELS_ROOT" ]] || { echo "SKIP: root not found: $MODELS_ROOT"; exit 0; }
[[ -d "$MODELS_ROOT/$LOADED_MODEL" ]] || { echo "SKIP: no checkpoint at $MODELS_ROOT/$LOADED_MODEL"; exit 0; }

trap 'pkill -9 -x mlx-serve 2>/dev/null; true' EXIT

# Test 1: --model + --model-dir → discovery enriches /v1/models
pkill -9 -x mlx-serve 2>/dev/null; sleep 1
"$BINARY" --model "$MODELS_ROOT/$LOADED_MODEL" --model-dir "$MODELS_ROOT" \
    --serve --port "$PORT" --ctx-size 4096 --log-level warn \
    --no-warmup-eager > /tmp/test_disc.log 2>&1 &
PID=$!
for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 0.5
    kill -0 "$PID" 2>/dev/null || { echo "FAIL: server died" >&2; exit 1; }
done

resp=$(curl -sf "http://127.0.0.1:$PORT/v1/models")
if [[ -z "$resp" ]]; then
    echo "FAIL: /v1/models empty" >&2
    exit 1
fi

echo "=== Test 1: explicit --model + --model-dir ==="
n_total=$(echo "$resp" | python3 -c "import json,sys;print(len(json.load(sys.stdin)['data']))")
n_loaded=$(echo "$resp" | python3 -c "import json,sys;print(sum(1 for m in json.load(sys.stdin)['data'] if m.get('loaded')))")
n_discoverable=$(echo "$resp" | python3 -c "import json,sys;print(sum(1 for m in json.load(sys.stdin)['data'] if not m.get('loaded')))")
echo "  total=$n_total loaded=$n_loaded discoverable=$n_discoverable"

if [[ "$n_total" -lt 2 ]]; then
    echo "FAIL: expected >=2 models in listing, got $n_total" >&2
    exit 1
fi
if [[ "$n_loaded" -ne 1 ]]; then
    echo "FAIL: expected exactly 1 loaded model, got $n_loaded" >&2
    exit 1
fi
if [[ "$n_discoverable" -lt 1 ]]; then
    echo "FAIL: expected >=1 discoverable sibling, got $n_discoverable" >&2
    exit 1
fi
echo "  PASS"

# Test 2: loaded model carries full meta; siblings carry bytes_on_disk
echo "=== Test 2: meta shape ==="
loaded_arch=$(echo "$resp" | python3 -c "
import json,sys
for m in json.load(sys.stdin)['data']:
    if m.get('loaded'):
        print(m.get('meta',{}).get('architecture','?'))
        break
")
sibling_bytes=$(echo "$resp" | python3 -c "
import json,sys
for m in json.load(sys.stdin)['data']:
    if not m.get('loaded'):
        b = m.get('meta',{}).get('bytes_on_disk')
        print(b if isinstance(b,(int,float)) else 0)
        break
")
echo "  loaded.architecture='$loaded_arch'  sibling.bytes_on_disk=$sibling_bytes"
if [[ -z "$loaded_arch" || "$loaded_arch" == "?" ]]; then
    echo "FAIL: loaded model missing architecture meta" >&2
    exit 1
fi
if [[ "$sibling_bytes" -le 0 ]]; then
    echo "FAIL: sibling missing bytes_on_disk" >&2
    exit 1
fi
echo "  PASS"

pkill -9 -x mlx-serve 2>/dev/null
sleep 1

# Test 3: --model-dir alone (no --model) auto-selects first discovered
echo "=== Test 3: --model-dir auto-select ==="
"$BINARY" --model-dir "$MODELS_ROOT" \
    --serve --port "$PORT" --ctx-size 4096 --log-level info \
    --no-warmup-eager > /tmp/test_disc2.log 2>&1 &
PID=$!
for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 0.5
    kill -0 "$PID" 2>/dev/null || { echo "FAIL: server died (auto-select path)" >&2; cat /tmp/test_disc2.log; exit 1; }
done

# --model-dir WITHOUT --model starts headless: no primary model is loaded and
# every discovery is registered as a stub that loads on demand. The old
# "Auto-selected --model" line was deliberately removed when that shipped
# (src/main.zig: "No auto-select ... the server starts HEADLESS"), so asserting
# it convicted the server of a bug it does not have.
loaded_count=$(curl -sf "http://127.0.0.1:$PORT/v1/models" \
    | grep -o '"loaded":true' | wc -l | tr -d ' ')
if [[ "$loaded_count" != "0" ]]; then
    echo "FAIL: headless boot loaded $loaded_count model(s); expected 0" >&2
    exit 1
fi
stub_count=$(curl -sf "http://127.0.0.1:$PORT/v1/models" \
    | grep -o '"loaded":false' | wc -l | tr -d ' ')
if [[ "$stub_count" -lt 1 ]]; then
    echo "FAIL: headless boot registered no stubs; expected the discovered siblings" >&2
    exit 1
fi
echo "  headless: 0 loaded, $stub_count stub(s) registered on demand"
echo "  PASS"

pkill -9 -x mlx-serve 2>/dev/null
echo
echo "=== ALL DISCOVERY TESTS PASSED ==="
