#!/usr/bin/env bash
# Offline smoke test for the embedded ds4 engine.
#
# Runs `mlx-serve --model <gguf> --prompt <text> --max-tokens N` and asserts
# the engine produces a non-empty, non-degenerate completion. NOT a byte-
# equivalence check — ds4 is the reference now.
#
# Server-mode integration is deferred (TODO #6 in the file at repo root); the
# offline path exercises the FFI, Metal kernel staging, tokenizer round-trip,
# and decode loop end-to-end without requiring the scheduler.
#
# Skips quietly if no ds4 GGUF is available locally so CI on smaller hosts
# doesn't fail. Set DS4_GGUF=<path> to point at a checkpoint.

set -euo pipefail

BINARY="${BINARY:-./zig-out/bin/mlx-serve}"

# Candidate paths, in order of preference.
DEFAULT_CANDIDATES=(
    "${DS4_GGUF:-}"
    "$HOME/.mlx-serve/models/antirez/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
    "$HOME/projects/agents/ds4/ds4flash.gguf"
    "$HOME/projects/agents/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
)

GGUF=""
for c in "${DEFAULT_CANDIDATES[@]}"; do
    if [[ -n "$c" && -f "$c" ]]; then
        GGUF="$c"
        break
    fi
done

if [[ -z "$GGUF" ]]; then
    echo "[skip] no ds4 GGUF found locally — set DS4_GGUF=<path> or drop one at ~/.mlx-serve/models/antirez/deepseek-v4-gguf/"
    exit 0
fi

if [[ ! -x "$BINARY" ]]; then
    echo "[fail] mlx-serve binary not found at $BINARY — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

echo "[ok] using GGUF: $GGUF"
echo "[ok] running prompt..."

OUTPUT=$("$BINARY" --model "$GGUF" --prompt "Write a haiku about Apple Silicon GPUs." --max-tokens 32 --temp 0.0 2>&1)

# Find the generated text — it follows the empty line after "[ds4] prompt: ... tokens".
GENERATED=$(printf '%s\n' "$OUTPUT" | awk 'flag && !/^\[ds4\]/ {print} /^\[ds4\] prompt:/ {flag=1}')

if [[ -z "$GENERATED" ]]; then
    echo "[fail] no generated text in output"
    printf '%s\n' "$OUTPUT" | tail -20
    exit 1
fi

# Reject degenerate output (any line that's just repeats of <｜begin▁of▁sentence｜>).
if printf '%s\n' "$GENERATED" | grep -qE '<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>'; then
    echo "[fail] degenerate output — model stuck on BOS"
    echo "generated: $GENERATED"
    exit 1
fi

GENERATED_WORDS=$(printf '%s\n' "$GENERATED" | wc -w | tr -d ' ')
if (( GENERATED_WORDS < 3 )); then
    echo "[fail] expected ≥3 words, got $GENERATED_WORDS"
    echo "generated: $GENERATED"
    exit 1
fi

echo "[ok] generated $GENERATED_WORDS words:"
echo "----"
printf '%s\n' "$GENERATED"
echo "----"
echo "[pass] tests/test_ds4_smoke.sh"
