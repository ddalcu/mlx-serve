#!/bin/bash
# One-shot setup for the three-way backend comparison.
#
# Downloads and unpacks the llama.cpp macOS arm64 server binary and the
# bartowski Qwen3.6-35B-A3B Q6_K GGUF weights into ~/.mlx-serve/backends/.
# Idempotent: skips any step whose artefact already exists.
#
# Sibling to:
#   ~/.mlx-serve/models/Qwen3.6-35B-A3B-6bit/       (MLX 6-bit, already present)
# Produces:
#   ~/.mlx-serve/backends/llama.cpp/bin/llama-server
#   ~/.mlx-serve/backends/gguf/Qwen_Qwen3.6-35B-A3B-Q6_K.gguf
#
# Usage: tests/compare_backends_setup.sh

set -u
set -o pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

LLAMA_TAG="b8839"
LLAMA_TARBALL="llama-${LLAMA_TAG}-bin-macos-arm64.tar.gz"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${LLAMA_TARBALL}"

GGUF_FILE="Qwen_Qwen3.6-35B-A3B-Q6_K.gguf"
GGUF_URL="https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/${GGUF_FILE}?download=true"

BACKENDS_ROOT="$HOME/.mlx-serve/backends"
LLAMA_DIR="$BACKENDS_ROOT/llama.cpp"
LLAMA_BIN="$LLAMA_DIR/bin/llama-server"
GGUF_DIR="$BACKENDS_ROOT/gguf"
GGUF_PATH="$GGUF_DIR/$GGUF_FILE"

MLX_MODEL_DIR="$HOME/.mlx-serve/models/Qwen3.6-35B-A3B-6bit"

say() { printf "${YELLOW}[setup]${NC} %s\n" "$*"; }
ok()  { printf "${GREEN}[ok]${NC}    %s\n" "$*"; }
die() { printf "${RED}[fail]${NC}  %s\n" "$*" >&2; exit 1; }

mkdir -p "$LLAMA_DIR" "$GGUF_DIR"

# ----- mlx-serve MLX weights (only verify presence) --------------------------
if [ ! -d "$MLX_MODEL_DIR" ]; then
    die "MLX weights missing: $MLX_MODEL_DIR. Download via the MLX Core app's Model Browser first."
fi
ok "MLX weights present: $MLX_MODEL_DIR"

# ----- llama.cpp binary ------------------------------------------------------
if [ -x "$LLAMA_BIN" ]; then
    ok "llama-server already installed: $LLAMA_BIN"
else
    say "Downloading llama.cpp $LLAMA_TAG macOS arm64 tarball"
    TMPDIR="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR"' EXIT
    if ! curl -fL --retry 3 --retry-delay 2 -C - -o "$TMPDIR/$LLAMA_TARBALL" "$LLAMA_URL"; then
        die "curl failed for $LLAMA_URL"
    fi
    say "Unpacking into $LLAMA_DIR"
    # Tarball layout: build/bin/llama-server, build/bin/libllama.dylib, etc.
    tar -xzf "$TMPDIR/$LLAMA_TARBALL" -C "$LLAMA_DIR" --strip-components=1 || \
        die "tar extraction failed"
    if [ ! -x "$LLAMA_BIN" ]; then
        # Some release tarballs nest the binary under build/bin/ — try a fallback locate.
        FOUND="$(find "$LLAMA_DIR" -type f -name llama-server -perm -u+x 2>/dev/null | head -1)"
        if [ -n "$FOUND" ] && [ "$FOUND" != "$LLAMA_BIN" ]; then
            mkdir -p "$LLAMA_DIR/bin"
            ln -sf "$FOUND" "$LLAMA_BIN"
        fi
    fi
    [ -x "$LLAMA_BIN" ] || die "llama-server not executable after extraction"
    ok "llama-server installed: $LLAMA_BIN"
fi

# Verify llama-server runs (--version) so Gatekeeper quarantine shows up now
if ! "$LLAMA_BIN" --version >/dev/null 2>&1; then
    say "llama-server failed first run. Attempting xattr clear (Gatekeeper quarantine)."
    xattr -dr com.apple.quarantine "$LLAMA_DIR" 2>/dev/null || true
    if ! "$LLAMA_BIN" --version >/dev/null 2>&1; then
        die "llama-server still refuses to run. Open it manually once in Finder to approve."
    fi
fi
ok "llama-server --version succeeds"

# ----- GGUF weights ----------------------------------------------------------
EXPECTED_GGUF_MIN_BYTES=$((28 * 1024 * 1024 * 1024))   # sanity floor ~28 GiB
if [ -f "$GGUF_PATH" ]; then
    sz=$(stat -f%z "$GGUF_PATH" 2>/dev/null || stat -c%s "$GGUF_PATH")
    if [ "$sz" -ge "$EXPECTED_GGUF_MIN_BYTES" ]; then
        ok "GGUF already present (${sz} bytes): $GGUF_PATH"
    else
        say "GGUF partial (${sz} bytes), resuming download"
        curl -fL --retry 3 --retry-delay 2 -C - -o "$GGUF_PATH" "$GGUF_URL" || \
            die "curl failed for $GGUF_URL"
    fi
else
    say "Downloading $GGUF_FILE (~30 GiB) — this takes a while"
    curl -fL --retry 3 --retry-delay 2 -C - -o "$GGUF_PATH" "$GGUF_URL" || \
        die "curl failed for $GGUF_URL"
fi

sz=$(stat -f%z "$GGUF_PATH" 2>/dev/null || stat -c%s "$GGUF_PATH")
if [ "$sz" -lt "$EXPECTED_GGUF_MIN_BYTES" ]; then
    die "GGUF size $sz below minimum $EXPECTED_GGUF_MIN_BYTES — download incomplete."
fi
ok "GGUF ready: $GGUF_PATH (${sz} bytes)"

# ----- mlx-lm (pip) ----------------------------------------------------------
if ! command -v mlx_lm.server >/dev/null 2>&1; then
    die "mlx_lm.server not on PATH. Install with: pip install mlx-lm"
fi
ok "mlx_lm.server on PATH: $(command -v mlx_lm.server)"

printf "${GREEN}All backends ready.${NC} Next: bash tests/compare_backends_run.sh\n"
