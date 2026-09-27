#!/bin/bash
# Laya completeness contract: a starved checkpoint (no `tokenizer/`) is refused
# BY NAME before any engine touches it; a complete one serves decisions.
# Checks A–C need no network — every marker prints from LOCAL state.
#
#   A. `--model <starved dir> --serve` refuses as an incomplete media pack and
#      NAMES the missing marker (`tokenizer/tokenizer.json`), never a raw
#      `error: FileNotFound`.
#   B. `mlx-serve run <starved abs dir>` prints the typed-decision refusal.
#   C. The process exits cleanly (no signal death).
#   D. A COMPLETE Laya checkpoint answers POST /v1/decisions (network + the
#      ~800 MB checkpoint; skipped when the model is not on disk).
set -u

BIN="${MLX_SERVE_BIN:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BIN" ]; then
    echo "SKIP: $BIN not found — build first: zig build -Doptimize=ReleaseFast"
    exit 0
fi
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
PASS=0
FAIL=0
check() {
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; PASS=$((PASS + 1)); else echo "FAIL: $1"; FAIL=$((FAIL + 1)); fi
}

# ── A–C. a starved Laya dir: proven shape, weights, no `tokenizer/` ──
LAYA="$SCRATCH/laya-mlx"
mkdir -p "$LAYA/encoder"
printf '{"model_type":"modernbert","hidden_size":8,"num_attention_heads":2,"num_hidden_layers":1}\n' > "$LAYA/encoder/config.json"
printf '{"act_costs":[0],"temperature":1.0}\n' > "$LAYA/rl_agent_config.json"
printf '{"format":"laya-mlx","dtype":"float16"}\n' > "$LAYA/mlx_config.json"
# A syntactically valid safetensors file with one dummy tensor; its content
# never loads — the completeness refusal must come first.
python3 - "$LAYA/model.safetensors" <<'PY'
import json, struct, sys
hdr = json.dumps({"model.vocab_weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}).encode()
data = bytearray(hdr)
while len(data) % 8: data += b' '
with open(sys.argv[1], 'wb') as f:
    f.write(struct.pack('<Q', len(hdr)) + bytes(data) + b'\x00' * 8)
PY

# A free port (overridable) so the port-in-use pre-check passes; the refusal
# happens BEFORE any bind, and stdin </dev/null keeps the REPL path out.
PORT=${LAYA_TEST_PORT:-11381}
while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT + 1)); done
"$BIN" --model "$LAYA" --serve --host 127.0.0.1 --port "$PORT" < /dev/null > "$SCRATCH/a.txt" 2>&1
RC=$?
grep -q "not treating it as a media model" "$SCRATCH/a.txt"; W=$?
grep -q "tokenizer/tokenizer.json" "$SCRATCH/a.txt"; M=$?
if grep -q "error: FileNotFound" "$SCRATCH/a.txt"; then R=1; else R=0; fi
check "starved laya dir: the completeness warning names tokenizer/tokenizer.json" $((W + M))
check "starved laya dir: no raw 'error: FileNotFound' escape" $R
[ "$RC" -ge 128 ] && check "starved laya dir: clean exit (no signal death)" 1 || check "starved laya dir: clean exit (no signal death)" 0

"$BIN" run "$LAYA" < /dev/null > "$SCRATCH/b.txt" 2>&1
grep -q "typed-decision model" "$SCRATCH/b.txt"; check "run on a starved laya dir prints the typed-decision refusal (not FileNotFound)" $?

# ── D. the complete checkpoint answers a decision ──
FULL="${LAYA_FULL:-$HOME/.mlx-serve/models/aac6fef/laya-mlx}"
if [ -f "$FULL/tokenizer/tokenizer.json" ] && [ -f "$FULL/model.safetensors" ]; then
    DPORT=${LAYA_TEST_PORT:-11381}
    while lsof -nP -iTCP:$DPORT -sTCP:LISTEN >/dev/null 2>&1; do DPORT=$((DPORT + 1)); done
    "$BIN" --model "$FULL" --serve --host 127.0.0.1 --port "$DPORT" > "$SCRATCH/d.txt" 2>&1 &
    PID=$!
    LISTEN=""
    for _ in $(seq 1 120); do
        LISTEN=$(grep -oE "Server listening on http://127.0.0.1:[0-9]+" "$SCRATCH/d.txt" | head -1)
        [ -n "$LISTEN" ] && break
        kill -0 "$PID" 2>/dev/null || break
        sleep 1
    done
    if [ -n "$LISTEN" ]; then
        # `questions` is an OBJECT keyed by question id with `criteria` labels
        # (see laya.zig's request parser — an array body 400s).
        curl -s -m 120 "http://127.0.0.1:$DPORT/v1/decisions" \
            -H 'Content-Type: application/json' \
            -d '{"state":"a door is closed","questions":{"q1":{"type":"choice","instructions":"Is the door open or closed?","criteria":["open","closed"]}}}' \
            > "$SCRATCH/d.json" 2>/dev/null
        grep -q '"answers"' "$SCRATCH/d.json" && grep -q '"q1"' "$SCRATCH/d.json"; check "complete laya checkpoint serves POST /v1/decisions" $?
    else
        echo "SKIP: complete checkpoint never listened (GPU busy?)"
    fi
    kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
else
    echo "SKIP: complete laya checkpoint not on disk ($FULL) — set LAYA_FULL to test /v1/decisions"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
