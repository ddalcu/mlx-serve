#!/usr/bin/env bash
# Build the Agent Sandbox guest kernel (arm64 Linux, virtiofs root) inside a
# Linux Docker container, ready to host as a GitHub release asset. Moved here
# from the contain repo (tools/build_kernel.sh) so the whole sandbox guest —
# rootfs image (../agent-shell-mlxserve) AND kernel — lives in one repo;
# mlx-serve is Apple-Silicon-only, so only the arm64 kernel is built.
#
# Produces:
#   artifacts/kernel-arm64             — raw Image (local boots: SANDBOX_KERNEL=...)
#   artifacts/release/kernel-arm64.gz  — the GitHub release asset
#   artifacts/release/SHA256SUMS
#
# Consumers (all pin the tag; bump together when republishing):
#   app/Sources/MLXServe/Services/AgentSandbox.swift  — kernelTag + kernelURL (DevID runtime fetch)
#   scripts/fetch-guest-rootfs.sh                     — KERNEL_TAG + KERNEL_SHA256 (MAS bundle staging)
#
# Release: create a GitHub release on ddalcu/mlx-serve named after the tag the
# consumers pin (e.g. kernels-v4) and upload artifacts/release/*.
#
# Patches in patches/*.patch are applied right after source extraction; a
# cached tree under .kbuild/ is assumed already patched, so after adding or
# changing a patch: rm -rf .kbuild/linux-*
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"

# 6.6.151: ≥6.6.69 is load-bearing on M4 — earlier kernels derive SVE hwcaps
# from the SME registers Apple exposes without checking FEAT_SVE, so guests
# advertise sve2 they can't execute and OpenSSL/Go SIGILL on the probe
# ("arm64: Filter out SVE hwcaps when FEAT_SVE isn't implemented").
VER="${KVER:-6.6.151}"
IMG="contain-kbuild:bookworm"
KBUILD="$HERE/.kbuild"          # source + object cache (gitignored)
OUT="$HERE/artifacts"
mkdir -p "$KBUILD" "$OUT/release"

command -v docker >/dev/null || { echo "need Docker (this script builds the kernel in a Linux container)"; exit 1; }

# --- one-time builder image (bookworm gcc-12 defaults to gnu17; newer gcc's
#     gnu23 default breaks parts of 6.6) ---
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "=== building kernel-builder image ($IMG) ==="
  docker build -t "$IMG" - <<'DOCKERFILE'
FROM debian:bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential bc bison flex libssl-dev libelf-dev \
      xz-utils curl ca-certificates cpio kmod \
  && rm -rf /var/lib/apt/lists/*
DOCKERFILE
fi

# Fetch + extract the kernel source on the HOST (the container may sit behind a
# TLS-intercepting proxy whose root CA it lacks), then apply our patches.
fetch_src() {
  [ -d "$KBUILD/linux-$VER" ] && return 0
  echo "=== downloading linux-$VER source (host) ==="
  curl -fSL "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-$VER.tar.xz" -o "$KBUILD/linux-$VER.tar.xz"
  tar -xf "$KBUILD/linux-$VER.tar.xz" -C "$KBUILD"
  rm -f "$KBUILD/linux-$VER.tar.xz"
  for p in "$HERE"/patches/*.patch; do
    [ -f "$p" ] || continue
    echo "=== applying $(basename "$p") ==="
    patch -d "$KBUILD/linux-$VER" -p1 < "$p"
  done
}

fetch_src
echo "=== building arm64 kernel (linux-$VER) ==="
docker run --rm -i -v "$KBUILD:/b" -v "$OUT:/out" -e VER="$VER" \
  -w /b "$IMG" bash -euo pipefail <<'INNER'
cd "/b/linux-$VER"
J="$(nproc)"

# defconfig already boots the VZ virt machine (PL011 + GICv2 + arch timer +
# virtio are =y). Add virtio-fs/vsock/net/rng + the container-runtime features
# real OCI workloads need, trim the big unused subsystems.
make ARCH=arm64 O=build-arm64 defconfig
m() { ./scripts/config --file build-arm64/.config "$@"; }
m --disable MODULES
m --enable  VIRTIO --enable VIRTIO_MMIO --enable VIRTIO_MMIO_CMDLINE_DEVICES
m --enable  VIRTIO_BLK --enable VIRTIO_NET
m --enable  HW_RANDOM --enable HW_RANDOM_VIRTIO
m --enable  NET_9P --enable NET_9P_VIRTIO --enable 9P_FS
# virtio-fs (FUSE): the rootfs + host-share transport. Non-DAX.
m --enable  FUSE_FS --enable VIRTIO_FS
m --disable FUSE_DAX
m --enable  OVERLAY_FS
# virtio-vsock: the vz-agent transport (AF_VSOCK).
m --enable  VSOCKETS --enable VIRTIO_VSOCKETS
# Container-runtime features real OCI workloads need.
m --enable  USER_NS
m --enable  MEMCG
m --enable  NETFILTER --enable NF_CONNTRACK --enable NF_NAT
m --enable  NF_TABLES --enable NF_TABLES_INET --enable NFT_CT --enable NFT_NAT --enable NFT_MASQ --enable NFT_COMPAT
m --disable NETFILTER_XT_TARGET_TCPMSS   # the one xt object whose O= Kbuild rule breaks
m --enable  SERIAL_AMBA_PL011 --enable SERIAL_AMBA_PL011_CONSOLE
m --enable  BLK_DEV_INITRD
# Trim defconfig hard for a headless virtio guest.
for o in ACPI DRM SOUND MEDIA_SUPPORT WLAN WIRELESS USB_SUPPORT INFINIBAND \
         SCSI ATA NVME_CORE MMC MTD MD BT NFC ETHERNET \
         HID IIO STAGING COMEDI NEW_LEDS HWMON THERMAL WATCHDOG POWER_SUPPLY \
         SND REGULATOR MEDIA_SUPPORT_FILTER CRYPTO_HW KEXEC CRASH_DUMP PROFILING \
         XFS_FS BTRFS_FS F2FS_FS GFS2_FS OCFS2_FS NILFS2_FS JFS_FS REISERFS_FS \
         NFS_FS NFSD CIFS CEPH_FS FAT_FS NTFS_FS HFS_FS HFSPLUS_FS UBIFS_FS \
         JFFS2_FS NUMA RANDOMIZE_BASE; do m --disable "$o"; done
# Re-assert the must-haves disabling a parent menu may have dropped.
m --enable EXT4_FS --enable OVERLAY_FS --enable TMPFS
m --enable NETDEVICES --enable VIRTIO_NET
make ARCH=arm64 O=build-arm64 olddefconfig
for s in CONFIG_FUSE_FS CONFIG_VIRTIO_FS CONFIG_OVERLAY_FS CONFIG_USER_NS CONFIG_MEMCG CONFIG_NF_TABLES CONFIG_VSOCKETS CONFIG_VIRTIO_VSOCKETS; do
  grep -q "^$s=y" build-arm64/.config || { echo "ERROR: $s not =y (olddefconfig dropped it)"; exit 1; }
done
make ARCH=arm64 O=build-arm64 -j"$J" Image
install -m644 build-arm64/arch/arm64/boot/Image /out/kernel-arm64
gzip -9 -c build-arm64/arch/arm64/boot/Image > /out/release/kernel-arm64.gz
ls -l /out/kernel-arm64 /out/release/kernel-arm64.gz
INNER

[ -f "$OUT/release/kernel-arm64.gz" ] || { echo "BUILD FAILED: release asset not produced"; exit 1; }

# The app's cache-validation byte-gates must hold on the shipped kernel.
for s in virtiofs virtio_vsock; do
  LC_ALL=C grep -qa "$s" "$OUT/kernel-arm64" || { echo "ERROR: '$s' not found in Image — AgentSandbox's kernelHas*Support gate would reject it"; exit 1; }
done

( cd "$OUT/release" && shasum -a 256 *.gz > SHA256SUMS && cat SHA256SUMS )

echo
echo "DONE. Upload artifacts/release/* to the ddalcu/mlx-serve GitHub release named"
echo "after the tag AgentSandbox.kernelTag / fetch-guest-rootfs.sh KERNEL_TAG pin."
