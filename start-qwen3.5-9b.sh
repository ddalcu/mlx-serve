#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-$HOME/.mlx-serve/models/Qwen3.5-9B-MLX-4bit}"

exec ./zig-out/bin/mlx-serve \
    --model "$MODEL_DIR" \
    --serve \
    --log-level info \
    "$@"
