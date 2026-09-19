#!/bin/bash
# A Prism Hadamard pack is served in its own numerics (f16 activations over
# its f16 scales, f32 GatedDeltaNet state): greedy logprobs must sit within
# the reference runtime's own fp16 distance of an f32 reference of the pack.
# A bf16 serve scored KL 1.7e-4 / top-1 98.3%; the f16 serve 2.9e-6 / 99.1%.
#
# Usage: HADAMARD_TEST_MODEL=<pack dir> ./tests/test_hadamard_fidelity.sh [port]
# Needs python3 with mlx, mlx_lm and tokenizers.
set -u
MODEL="${HADAMARD_TEST_MODEL:-/Volumes/G Drive SSD/models-dl/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit}"
PORT="${1:-11281}"
BIN="${BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -d "$MODEL" ]; then echo "skip: model not found ($MODEL)"; exit 0; fi

LOG=$(mktemp)
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --no-mtp --log-level info >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f "$LOG"' EXIT
for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done

OUT=$(python3 "$(dirname "$0")/hadamard_fidelity.py" "$MODEL" "$PORT") || { echo "FAIL: fidelity run"; exit 1; }
echo "[hadamard-fidelity] $OUT"
eval "$(echo "$OUT" | tr ' ' '\n')"
python3 -c "import sys; sys.exit(0 if float('$KL') < 1e-5 and float('$TOP1') >= 98.5 else 1)" \
    && echo "PASS: within the reference's fp16 distance" \
    || { echo "FAIL: KL $KL / top-1 $TOP1% (want < 1e-5 and >= 98.5%)"; exit 1; }
