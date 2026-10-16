#!/usr/bin/env bash
# Fetch the pinned Zig release and stage it at .zig-toolchain/ (stable path,
# independent of the version string in the tarball's own top-level dir name).
#
# Pinned to Zig 0.17.0 stable. 0.16.0's bundled libc++ fails to compile
# against the macOS 27 SDK (`use of undeclared identifier 'INFINITY'` in its
# vendored <random> — see build.zig's version-gate comptime block).
#
# This is the single source of truth for the pinned Zig version. Bump
# ZIG_VERSION to upgrade; CI and local builds re-fetch automatically.
set -euo pipefail

ZIG_VERSION="${ZIG_VERSION:-0.17.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/.zig-toolchain"
STAMP="$DEST/.version"

case "$(uname -s)" in
  Darwin) OS="macos" ;;
  Linux) OS="linux" ;;
  MINGW*|MSYS*|CYGWIN*) OS="windows" ;; # Git Bash
  *) echo "[fetch-zig] ERROR: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac
if [ "$OS" = windows ]; then EXE=zig.exe; EXT=zip; else EXE=zig; EXT=tar.xz; fi

# Idempotent: skip when the staged copy already matches the pinned version.
if [ -f "$STAMP" ] && [ -x "$DEST/$EXE" ]; then
  if [ "$(cat "$STAMP")" = "$ZIG_VERSION" ]; then
    echo "[fetch-zig] .zig-toolchain already at $ZIG_VERSION — nothing to do"
    exit 0
  fi
  echo "[fetch-zig] staged version '$(cat "$STAMP")' != '$ZIG_VERSION' — refetching"
  # The build cache holds configure-time paths from the old toolchain.
  rm -rf "$REPO_ROOT/.zig-cache"
fi

case "$(uname -m)" in
  arm64|aarch64) ARCH="aarch64" ;; # macOS says arm64, Linux aarch64
  x86_64) ARCH="x86_64" ;;
  *) echo "[fetch-zig] ERROR: unsupported arch $(uname -m)" >&2; exit 1 ;;
esac
ASSET="zig-${ARCH}-${OS}-${ZIG_VERSION}.${EXT}"
URL="https://ziglang.org/download/${ZIG_VERSION}/${ASSET}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[fetch-zig] downloading $URL"
curl -fSL --retry 3 -o "$TMP/$ASSET" "$URL"

echo "[fetch-zig] extracting"
if [ "$EXT" = zip ]; then unzip -q "$TMP/$ASSET" -d "$TMP"; else tar xf "$TMP/$ASSET" -C "$TMP"; fi

EXTRACTED="$TMP/zig-${ARCH}-${OS}-${ZIG_VERSION}"
if [ ! -x "$EXTRACTED/$EXE" ]; then
  echo "[fetch-zig] ERROR: no zig executable in $ASSET" >&2
  exit 1
fi

rm -rf "$DEST"
mv "$EXTRACTED" "$DEST" # a rename on one volume: Git Bash's cp takes minutes over Zig's ~16k files

echo "$ZIG_VERSION" > "$STAMP"

echo "[fetch-zig] staged Zig ($ZIG_VERSION):"
echo "  $DEST/$EXE ($("$DEST/$EXE" version))"
echo ""
echo "  Add it to PATH for this shell:"
echo "    export PATH=\"$DEST:\$PATH\""
