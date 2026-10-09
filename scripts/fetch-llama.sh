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
# On Linux it stages the release's CUDA build instead: libllama + the ggml
# backend plugins (libggml-cuda, libggml-cpu-*) it loads from beside itself, and
# the headers from the tagged source (the Linux archive ships none).
#
# This is the single source of truth for the pinned llama.cpp version.
# Bump LLAMA_TAG to upgrade; CI and local builds re-fetch automatically.
set -euo pipefail

# Release v0.6.0 is commit b11429; its tag ships no binaries, the build tag does.
LLAMA_TAG="${LLAMA_TAG:-b11429}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/lib/llama"
DEST_LIB="$DEST/lib"
DEST_INC="$DEST/include"
STAMP="$DEST/.version"
OS="$(uname -s)"
if [ "$OS" = Linux ]; then LIB=libllama.so; else LIB=libllama.dylib; fi

# Idempotent: skip when the staged copy already matches the pinned tag.
if [ -f "$STAMP" ] && [ -f "$DEST_LIB/$LIB" ] && [ -f "$DEST_INC/llama.h" ]; then
  if [ "$(cat "$STAMP")" = "$LLAMA_TAG" ]; then
    echo "[fetch-llama] lib/llama already at $LLAMA_TAG — nothing to do"
    exit 0
  fi
  echo "[fetch-llama] staged version '$(cat "$STAMP")' != '$LLAMA_TAG' — refetching"
fi

RELEASE="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ "$OS" = Linux ]; then
  case "$(uname -m)" in x86_64) ARCH=x64 ;; aarch64) ARCH=arm64 ;; *) echo "[fetch-llama] ERROR: unsupported arch $(uname -m)" >&2; exit 1 ;; esac
  URL="$RELEASE/llama-${LLAMA_TAG}-bin-ubuntu-cuda-13.4-${ARCH}.tar.gz"
  echo "[fetch-llama] downloading $URL"
  curl -fSL --retry 3 -o "$TMP/bin.tgz" "$URL"
  curl -fSL --retry 3 -o "$TMP/src.tgz" "https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_TAG}.tar.gz"
  mkdir -p "$TMP/bin" "$TMP/src"
  tar -xzf "$TMP/bin.tgz" -C "$TMP/bin"
  tar -xzf "$TMP/src.tgz" -C "$TMP/src" --strip-components=1 --wildcards '*/include/*.h'
  rm -rf "$DEST_LIB" "$DEST_INC"
  mkdir -p "$DEST_LIB" "$DEST_INC"
  find "$TMP/bin" \( -name 'libllama.so*' -o -name 'libggml*.so*' \) -exec cp -P {} "$DEST_LIB/" \;
  cp "$TMP/src/include/"*.h "$TMP/src/ggml/include/"*.h "$DEST_INC/"
  echo "$LLAMA_TAG" > "$STAMP"
  echo "[fetch-llama] staged libllama ($LLAMA_TAG, CUDA) in $DEST_LIB"
  exit 0
fi

URL="$RELEASE/llama-${LLAMA_TAG}-xcframework.zip"
echo "[fetch-llama] downloading $URL"
curl -fSL --retry 3 -o "$TMP/xcf.zip" "$URL"

echo "[fetch-llama] extracting macOS slice"
unzip -q "$TMP/xcf.zip" -d "$TMP/xcf"

FW="$(find "$TMP/xcf" -type d -path '*macos-arm64*/llama.framework' | head -1)"
if [ -z "$FW" ]; then
  echo "[fetch-llama] ERROR: no macos-arm64 llama.framework in $URL" >&2
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
