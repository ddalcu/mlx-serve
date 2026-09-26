#!/usr/bin/env bash
# Fetch the pinned sushi release: the guest engine mlx-serve runs as a separate
# process for Qwen3.8-Flash-Next EXL3 packs (src/arch/sushi_guest.zig).
#
# This is the single source of truth for the pin: build.zig reads SUSHI_TAG,
# SUSHI_SHA256 (the tarball) and SUSHI_TREE_SHA256 (the unpacked tree, see
# tree_sha) from the three lines below. The server downloads the same tarball on
# first use into the same folder and re-checks the tree before every launch;
# running this script stages it ahead of time. The tarball is checked before
# anything is extracted, and the tree before it is moved into place.
set -euo pipefail

SUSHI_TAG="${SUSHI_TAG:-v1.0.3}"
SUSHI_SHA256="${SUSHI_SHA256:-a317a9e44a98c052b25eb7fb5f8f0694d9c34931b0cea731b46f0faa4cc72d79}"
SUSHI_TREE_SHA256="${SUSHI_TREE_SHA256:-a6ceae0508e1637ef0f1411d85511e8994985e967f49d76cf0201dc4d8ffe464}"

ASSET="sushi-bin-macos-arm64.tar.gz"
URL="https://github.com/beamivalice/sushi/releases/download/${SUSHI_TAG}/${ASSET}"
DEST="$HOME/.mlx-serve/engines/sushi/${SUSHI_TAG}"
BIN="$DEST/sushi-macos-arm64/sushi"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
# sha256 over the sorted "<relative path>\0<sha256>\n" lines of every regular file
# (the server's treeSha256Hex computes the same).
tree_sha() {
  (cd "$1" && find . -type f | sed 's|^\./||' | LC_ALL=C sort |
    while IFS= read -r f; do printf '%s\0%s\n' "$f" "$(sha "$f")"; done) | sha /dev/stdin
}

if [ -d "$DEST/sushi-macos-arm64" ] && [ "$(tree_sha "$DEST/sushi-macos-arm64")" = "$SUSHI_TREE_SHA256" ]; then
  echo "[fetch-sushi] $DEST already at $SUSHI_TAG — nothing to do"
  exit 0
fi

STAGING="$DEST.partial"
rm -rf "$STAGING"
mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT

echo "[fetch-sushi] downloading $URL"
curl -fSL --retry 3 --connect-timeout 30 --speed-limit 10000 --speed-time 60 -o "$STAGING/$ASSET" "$URL"

GOT="$(sha "$STAGING/$ASSET")"
if [ "$GOT" != "$SUSHI_SHA256" ]; then
  echo "[fetch-sushi] ERROR: sha256 $GOT, pinned $SUSHI_SHA256" >&2
  exit 1
fi

tar -xzf "$STAGING/$ASSET" -C "$STAGING"
rm -f "$STAGING/$ASSET"
GOT="$(tree_sha "$STAGING/sushi-macos-arm64")"
if [ "$GOT" != "$SUSHI_TREE_SHA256" ]; then
  echo "[fetch-sushi] ERROR: tree sha256 $GOT, pinned $SUSHI_TREE_SHA256" >&2
  exit 1
fi
# A stale or tampered tree is replaced; mv would nest the new one inside it.
rm -rf "$DEST"
mv "$STAGING" "$DEST"
trap - EXIT

echo "[fetch-sushi] staged sushi $SUSHI_TAG:"
"$BIN" --guest-manifest 2>/dev/null
