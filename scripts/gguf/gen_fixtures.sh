#!/bin/bash
# Regenerates src/sdk/quant/gguf/fixtures/*.bin from ggml's reference dequant.
# Needs a libllama.dylib (it exports dequantize_row_*), default: the one staged in mlx-serve.
set -euo pipefail
cd "$(dirname "$0")/../.."
LLAMA_LIB="${LLAMA_LIB:-lib/llama/lib}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cc -O0 -o "$tmp/gen_fixtures" scripts/gguf/gen_fixtures.c -L"$LLAMA_LIB" -lllama -Wl,-rpath,"$(cd "$LLAMA_LIB" && pwd)"
"$tmp/gen_fixtures" src/sdk/quant/gguf/fixtures
ls -la src/sdk/quant/gguf/fixtures
