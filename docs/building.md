# Building from source

You only need this if you're hacking on mlx-serve. To just use it, grab [the app](https://github.com/ddalcu/mlx-serve/releases/latest) or `brew install mlx-serve`.

## Prerequisites

- macOS 26.2+ with Apple Silicon (M1/M2/M3/M4/M5) — the bundled MLX is built at deployment target 26.2 so the M5 neural-accelerator (NAX) kernels ship enabled
- [Zig 0.17 nightly](https://ziglang.org/download/) — staged automatically by `./scripts/fetch-zig.sh` into `.zig-toolchain/`
- libwebp: `brew install webp`
- Xcode 26.2+ with the Metal Toolchain component — mlx + mlx-c are pinned submodules compiled by `scripts/build-mlx.sh`, not brew packages, so the NAX kernels the brew bottle silently omits are included

## App + server

One script builds everything:

```bash
git clone --recurse-submodules https://github.com/ddalcu/mlx-serve && cd mlx-serve
APPLE_DEVELOPER_ID=- APPLE_TEAM_ID=- SKIP_NOTARIZE=1 ./app/build.sh
open "app/MLX Core.app"
```

`app/build.sh` snaps the pinned submodules back to their commits, stages llama.cpp and the Zig nightly, builds mlx + mlx-c with NAX kernels asserted, compiles the Swift app and the Zig server, then bundles and signs. `APPLE_DEVELOPER_ID=-` picks ad-hoc signing, so no Apple developer account is needed. A notarized release build wants a real identity plus `APPLE_ID` and `APPLE_ID_PASSWORD`.

## Server only

```bash
./scripts/fetch-llama.sh && ./scripts/build-mlx.sh   # once, and again on a pin bump
zig build -Doptimize=ReleaseFast                     # always ReleaseFast; Debug is 2-4x slower
```
