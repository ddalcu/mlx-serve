# Agent Sandbox guest kernel

The arm64 Linux kernel the Agent Sandbox guest boots (virtiofs root, vsock
agent transport). Built from a pinned upstream release plus our patches;
published as a GitHub release asset on this repo that both app builds consume:

- **Developer ID**: downloaded at runtime — `AgentSandbox.kernelTag` /
  `kernelURL` (`app/Sources/MLXServe/Services/AgentSandbox.swift`).
- **Mac App Store**: staged into the bundle — `KERNEL_TAG` / `KERNEL_SHA256`
  in `scripts/fetch-guest-rootfs.sh`.

The rootfs half of the guest lives next door in `../agent-shell-mlxserve/`.
The kernel was previously built in and released from the `contain` repo
(`kernels-v2`/`v3` stay published there for old caches); from `kernels-v4` on
it lives here.

## Patches

- `0001-fuse-virtiofs-owner-read-create.patch` — issue #150. Apple's
  Virtualization.framework virtiofs device runs host-side as an unprivileged
  process, so any inode whose mode lacks owner-read is a black hole for the
  guest: creates fail EACCES (GNU tar's dangling-symlink placeholders, dpkg's
  `.dpkg-new` staging files — i.e. `apt-get install` was broken wholesale),
  and an existing one can't be stat'ed, opened, chmod'ed back, or unlinked.
  The patch adds a virtiofs-only fuse connection flag that ORs owner-read
  into create/mkdir/chmod modes (directories get owner-rwx — the host also
  needs write+search to create and look up children). Semantics deviation is
  deliberate and tiny: `chmod 000 f` lands as 400 — in exchange, root in the
  guest can install and write anything, like a normal Linux machine.

## Building + releasing

```
./build.sh                 # needs Docker; produces artifacts/release/*
```

Then create a GitHub release on ddalcu/mlx-serve named after the new tag
(e.g. `kernels-v5`), upload `artifacts/release/kernel-arm64.gz` +
`SHA256SUMS`, and bump the two consumer pins above in the same change.
A stale cached tree ignores new patches — `rm -rf .kbuild/linux-*` first.
