#!/bin/bash
# Quick benchmark script for mlx-serve performance testing
# Usage: ./bench.sh [binary_path] [model_path]

BINARY="${1:-./zig-out/bin/mlx-serve}"
MODEL="${2:-$HOME/.mlx-serve/models/gemma-4-e4b-it-4bit}"
PROMPT="Explain quantum computing in 200 words"
MAX_TOKENS=256
RUNS=3

echo "=== mlx-serve Benchmark ==="
echo "Binary: $BINARY"
echo "Model: $(basename $MODEL)"
echo "Max tokens: $MAX_TOKENS"
echo "Runs: $RUNS"
echo ""

for i in $(seq 1 $RUNS); do
    echo "--- Run $i ---"
    $BINARY --model "$MODEL" --prompt "$PROMPT" --max-tokens $MAX_TOKENS 2>&1 | grep -E "^(Prompt:|Generation:|Peak memory:)"
    echo ""
done
