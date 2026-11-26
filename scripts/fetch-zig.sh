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
#
# On macOS the staged lib/ carries the patches/zig-<version>-*.patch files
# (std fixes not yet upstream; each patch says what it fixes). Their SHA-1 is
# the stamp's patch=, so a copy staged without them, or with another set, is
# brought up to date: in place when it is the plain release of the same
# version, else by a refetch.
#
#   scripts/fetch-zig.sh           stage (or bring up to date) .zig-toolchain/
#   scripts/fetch-zig.sh --check   exit 0 when the staged copy matches, else
#                                  say why and exit 1; changes nothing
set -euo pipefail

ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
CHECK=0
case "${1:-}" in
  "") ;;
  --check) CHECK=1 ;;
  *) echo "[fetch-zig] usage: fetch-zig.sh [--check]" >&2; exit 2 ;;
esac

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

# The patches for this version (macOS: the code they touch is Darwin's), in name order, and the stamp they give.
PATCHES=()
if [ "$OS" = macos ]; then
  for p in "$REPO_ROOT"/patches/zig-"$ZIG_VERSION"-*.patch; do [ -f "$p" ] && PATCHES+=("$p"); done
fi
WANT="$ZIG_VERSION"
if [ ${#PATCHES[@]} -gt 0 ]; then WANT="$ZIG_VERSION patch=$(cat "${PATCHES[@]}" | shasum | cut -c1-12)"; fi

# apply_patches <toolchain dir>: every patch checked first, then applied; a patch that does not apply stops the script.
apply_patches() {
  local p
  for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
    patch -p1 -s -N -C -d "$1" < "$p" > /dev/null || { echo "[fetch-zig] ERROR: $p does not apply to $1" >&2; exit 1; }
  done
  for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
    patch -p1 -s -N -d "$1" < "$p" || { echo "[fetch-zig] ERROR: applying $p to $1 failed" >&2; exit 1; }
    echo "[fetch-zig] applied $(basename "$p")"
  done
}

HAVE=""
if [ -f "$STAMP" ] && [ -x "$DEST/$EXE" ]; then HAVE="$(cat "$STAMP")"; fi
if [ "$CHECK" = 1 ]; then
  if [ "$HAVE" = "$WANT" ]; then echo "[fetch-zig] .zig-toolchain is at '$WANT'"; exit 0; fi
  echo "[fetch-zig] .zig-toolchain is at '${HAVE:-nothing staged}', want '$WANT': run scripts/fetch-zig.sh"
  exit 1
fi

# Idempotent: skip when the staged copy already matches the pinned version and patches.
if [ -n "$HAVE" ]; then
  if [ "$HAVE" = "$WANT" ]; then
    echo "[fetch-zig] .zig-toolchain already at $WANT — nothing to do"
    exit 0
  fi
  if [ "$HAVE" = "$ZIG_VERSION" ]; then
    # The plain release of this version: patch it in place (patch replaces each file by a rename).
    echo "[fetch-zig] staged '$HAVE' lacks the patches — applying them"
    apply_patches "$DEST"
    echo "$WANT" > "$STAMP"
    echo "[fetch-zig] .zig-toolchain now at $WANT"
    exit 0
  fi
  echo "[fetch-zig] staged version '$HAVE' != '$WANT' — refetching"
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
apply_patches "$EXTRACTED"

rm -rf "$DEST"
mv "$EXTRACTED" "$DEST" # a rename on one volume: Git Bash's cp takes minutes over Zig's ~16k files

echo "$WANT" > "$STAMP"

echo "[fetch-zig] staged Zig ($WANT):"
echo "  $DEST/$EXE ($("$DEST/$EXE" version))"
echo ""
echo "  Add it to PATH for this shell:"
echo "    export PATH=\"$DEST:\$PATH\""
