#!/usr/bin/env bash
# Stage the bundled Agent Sandbox guest (kernel + rootfs) 
#
# Two assets:
#   * kernel        — contain's prebuilt kernel (GH release, pinned by tag + SHA256)
#   * rootfs.tar.gz — `docker export` of the SAME Docker Hub image the Developer
#                     ID build's sandbox pulls at runtime (ddalcu/agent-shell-mlxserve),
#                     pinned by CONTENT DIGEST so a re-tagged `:latest` can't
#                     silently change what ships. One image, both builds.
#
# Requires Docker (the export needs a container filesystem merge — layers,
# whiteouts — which `docker export` does canonically). This runs on a dev Mac
# for local MAS builds; release.yml never calls it.
#
# This is the single source of truth for the bundled-guest versions. Bump the
# tag/digest to upgrade; the App Store build re-stages automatically.
set -euo pipefail

KERNEL_TAG="${GUEST_KERNEL_TAG:-kernels-v3}"
KERNEL_REPO="${GUEST_KERNEL_REPO:-ddalcu/contain}"
ROOTFS_IMAGE="${GUEST_ROOTFS_IMAGE:-ddalcu/agent-shell-mlxserve}"
# Manifest-list digest of the pinned image (docker buildx imagetools inspect).
# Empty = pull `:latest` unpinned — first-bring-up escape hatch only; a build
# from an unpinned tag is not reproducible.
ROOTFS_DIGEST="${GUEST_ROOTFS_DIGEST:-sha256:a46f6170612a29828cc2567f337e2c28a85981f7d0068f02a6937cfafb0c4db0}"

# Expected kernel SHA256 (of the .gz release asset). The rootfs needs no
# separate hash — the docker digest IS content-addressed.
KERNEL_SHA256="${GUEST_KERNEL_SHA256:-d312d9cd9d18c9dfeebe87a9b8441742b1595e664c1ddf59c9096be544b6d0f1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/lib/guest"
STAMP="$DEST/.version"
WANT="$KERNEL_TAG:${ROOTFS_DIGEST:-unpinned}"

mkdir -p "$DEST"

# Idempotent: skip when the staged copy already matches the pinned versions.
# An unpinned (:latest) stage is NEVER treated as current — it re-exports.
if [ -n "$ROOTFS_DIGEST" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ] \
   && [ -f "$DEST/kernel" ] && [ -f "$DEST/rootfs.tar.gz" ]; then
    echo "guest assets already staged ($WANT)"
    exit 0
fi

verify() { # path expected-sha256 name
    local path="$1" want="$2" name="$3"
    [ -n "$want" ] || { echo "  ($name checksum not pinned — skipping verify)"; return 0; }
    local got
    got="$(shasum -a 256 "$path" | awk '{print $1}')"
    if [ "$got" != "$want" ]; then
        echo "ERROR: $name checksum mismatch" >&2
        echo "  expected $want" >&2
        echo "  got      $got" >&2
        exit 1
    fi
    echo "  $name checksum OK"
}

echo "→ Fetching guest kernel ($KERNEL_REPO@$KERNEL_TAG)..."
curl -fL --retry 3 --retry-delay 2 -o "$DEST/kernel.gz" \
    "https://github.com/$KERNEL_REPO/releases/download/$KERNEL_TAG/kernel-contain-arm64.gz"
verify "$DEST/kernel.gz" "$KERNEL_SHA256" "kernel.gz"
gunzip -f "$DEST/kernel.gz" # -> $DEST/kernel

REF="$ROOTFS_IMAGE${ROOTFS_DIGEST:+@$ROOTFS_DIGEST}"
echo "→ Exporting guest rootfs ($REF)..."
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running — the rootfs is exported from the $ROOTFS_IMAGE image" >&2
    exit 1
fi
[ -n "$ROOTFS_DIGEST" ] || echo "  WARNING: GUEST_ROOTFS_DIGEST is empty — exporting an UNPINNED :latest (not reproducible)"

# The guest is ALWAYS arm64 (Apple Silicon VM); letting docker default to the
# host platform is fine on an arm64 Mac but --platform makes it explicit.
docker pull --platform linux/arm64 "$REF" >/dev/null

CONTAINER="mlxserve-rootfs-export-$$"
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT
docker create --platform linux/arm64 --name "$CONTAINER" "$REF" >/dev/null
docker export "$CONTAINER" | gzip -9 > "$DEST/rootfs.tar.gz"
docker rm "$CONTAINER" >/dev/null
trap - EXIT

# Sanity: an empty/botched export would otherwise surface as a cryptic
# provisioner error inside the sandboxed app.
if ! gzip -t "$DEST/rootfs.tar.gz" || [ "$(stat -f%z "$DEST/rootfs.tar.gz")" -lt 10000000 ]; then
    echo "ERROR: exported rootfs.tar.gz is corrupt or implausibly small" >&2
    exit 1
fi

echo "$WANT" > "$STAMP"
echo "  Staged: $DEST/{kernel, rootfs.tar.gz} ($(du -h "$DEST/rootfs.tar.gz" | cut -f1 | tr -d ' ') rootfs)"
