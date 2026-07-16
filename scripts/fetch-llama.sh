#!/usr/bin/env bash
# Fetch llama.cpp's prebuilt libllama (the inference library, NOT llama-server)
# and stage it for linking into the mlx-serve Zig binary.
#
# The XCFramework ships libllama as a single self-contained dylib
# (llama + ggml + ggml-metal merged, Metal shaders embedded) that depends only
# on system frameworks. We thin it to arm64 (the app is Apple-Silicon only),
# rewrite its install-name to @rpath/libllama.dylib, and drop the headers next
# to it. build.zig links against lib/llama, and release.yml / app/build.sh
# bundle + re-sign the dylib exactly like libmlxc.dylib.
#
# This is the single source of truth for the pinned llama.cpp version.
# Bump LLAMA_TAG to upgrade; CI and local builds re-fetch automatically.
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b10034}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/lib/llama"
DEST_LIB="$DEST/lib"
DEST_INC="$DEST/include"
STAMP="$DEST/.version"

# Idempotent: skip when the staged copy already matches the pinned tag.
if [ -f "$STAMP" ] && [ -f "$DEST_LIB/libllama.dylib" ] && [ -f "$DEST_INC/llama.h" ]; then
  if [ "$(cat "$STAMP")" = "$LLAMA_TAG" ]; then
    echo "[fetch-llama] lib/llama already at $LLAMA_TAG — nothing to do"
    exit 0
  fi
  echo "[fetch-llama] staged version '$(cat "$STAMP")' != '$LLAMA_TAG' — refetching"
fi

ASSET="llama-${LLAMA_TAG}-xcframework.zip"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${ASSET}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[fetch-llama] downloading $URL"
curl -fSL --retry 3 -o "$TMP/xcf.zip" "$URL"

echo "[fetch-llama] extracting macOS slice"
unzip -q "$TMP/xcf.zip" -d "$TMP/xcf"

FW="$(find "$TMP/xcf" -type d -path '*macos-arm64*/llama.framework' | head -1)"
if [ -z "$FW" ]; then
  echo "[fetch-llama] ERROR: no macos-arm64 llama.framework in $ASSET" >&2
  exit 1
fi
FW_BIN="$FW/Versions/A/llama"
FW_HEADERS="$FW/Versions/A/Headers"

rm -rf "$DEST_LIB" "$DEST_INC"
mkdir -p "$DEST_LIB" "$DEST_INC"

# Thin the universal framework binary to arm64 (falls back to a copy if it is
# already single-arch), then expose it as a conventionally-named dylib.
if lipo -archs "$FW_BIN" 2>/dev/null | grep -q x86_64; then
  lipo -thin arm64 "$FW_BIN" -output "$DEST_LIB/libllama.dylib"
else
  cp "$FW_BIN" "$DEST_LIB/libllama.dylib"
fi

# Rewrite the framework-style install-name to a plain @rpath dylib so the
# linker and our bundle-time install_name_tool rewrites (release.yml / build.sh)
# can treat it like any other bundled dylib.
install_name_tool -id "@rpath/libllama.dylib" "$DEST_LIB/libllama.dylib"

# Re-sign ad-hoc: install_name_tool invalidates the signature, and dyld refuses
# to load an arm64 dylib with a stale signature. Bundle steps re-sign with the
# Developer ID later.
codesign --remove-signature "$DEST_LIB/libllama.dylib" 2>/dev/null || true
codesign --force --sign - "$DEST_LIB/libllama.dylib"

cp "$FW_HEADERS"/*.h "$DEST_INC/"

echo "$LLAMA_TAG" > "$STAMP"

echo "[fetch-llama] staged libllama ($LLAMA_TAG):"
echo "  $DEST_LIB/libllama.dylib ($(du -h "$DEST_LIB/libllama.dylib" | cut -f1))"
echo "  $(ls "$DEST_INC" | wc -l | tr -d ' ') headers in $DEST_INC"
