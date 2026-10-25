[English](building.md) · [简体中文](zh-CN/building.md)

# Building from source

You only need this if you're hacking on mlx-serve. To just use it, grab [the app](https://github.com/ddalcu/mlx-serve/releases/latest) or `brew install mlx-serve`.

## Prerequisites

- macOS 26.2+ with Apple Silicon (M1/M2/M3/M4/M5) — the bundled MLX is built at deployment target 26.2 so the M5 neural-accelerator (NAX) kernels ship enabled
- Xcode 26.2+ with the Metal Toolchain component — mlx + mlx-c are pinned submodules compiled by `scripts/build-mlx.sh`, not brew packages, so the NAX kernels the brew bottle silently omits are included. Xcode 26 ships the Metal compiler as a separate download, so if `xcrun -sdk macosx metal --version` fails, run `xcodebuild -downloadComponent MetalToolchain` first
- cmake and libwebp: `brew bundle install --file=Brewfile` from the repo root. cmake builds the mlx submodules, webp decodes images in the vision pipeline (`webp >= 1.6.0`, checked at build time)
- [Zig 0.17](https://ziglang.org/download/) — staged automatically by `./scripts/fetch-zig.sh` into `.zig-toolchain/`

## App + server

One script builds everything:

```bash
git clone --recurse-submodules https://github.com/ddalcu/mlx-serve && cd mlx-serve
brew bundle install --file=Brewfile
./app/build.sh
open "app/MLX-Serve.app"
```

`app/build.sh` snaps the pinned submodules back to their commits, stages llama.cpp and Zig, builds mlx + mlx-c with NAX kernels asserted, compiles the Swift app and the Zig server, then bundles and signs. With no signing identity in the environment it signs ad-hoc and skips notarization, so no Apple developer account is needed. Releases are cut by `.github/workflows/release.yml`.

## Server only

```bash
./scripts/fetch-zig.sh                               # stages the pinned Zig at .zig-toolchain/
export PATH="$PWD/.zig-toolchain:$PATH"
./scripts/fetch-llama.sh && ./scripts/build-mlx.sh   # once, and again on a pin bump
zig build -Doptimize=ReleaseFast                     # always ReleaseFast; Debug is 2-4x slower
```

## Linux and Windows (NVIDIA)

Every release attaches `mlx-serve-linux-x64-cuda13.tar.gz` (RTX 20 to 50 series) and `mlx-serve-linux-arm64-cuda13.tar.gz` (GH200, Jetson Thor, DGX Spark). They need an NVIDIA driver 580 or newer, the CUDA 13 toolkit, cuDNN 9, OpenBLAS/LAPACKE, libwebp and the Avahi dns_sd compat library. Unpack into an empty folder (the tarball has no top-level directory) and run `zig-out/bin/mlx-serve`; it finds its libraries next to itself.

To build on Linux, run `./scripts/build-linux.sh` from a fresh checkout. On Windows, `powershell -File scripts\build-windows.ps1` builds `mlx-serve.exe` into `dist\mlx-serve-windows-x64.zip`; it needs the Visual Studio 2022 C++ tools, CMake, Ninja, Git, the CUDA 13.0 toolkit and cuDNN 9, and running it needs an NVIDIA driver with CUDA 13 and cuDNN 9. Kernels, per-device defaults and memory: [CUDA.md](CUDA.md).

## Hermetic tests (Linux too)

The per-step preview encoder (`src/preview.zig` + `src/jpeg.zig` + `src/latent_rgb.zig`) links no MLX and no Homebrew webp, so any machine, a Linux Cloud Agent included, can build and run it with no GPU:

```bash
./scripts/fetch-zig.sh
export PATH="$PWD/.zig-toolchain:$PATH"
zig build preview-test
```

The same step exists on a Mac and builds the same hermetic artifact, but it is **not** a way to build without a staged mlx: `verifyBrewDeps` and `verifyMlxStage` run at configure time for every step, so `lib/mlx/` must already be built. On a Mac `zig build test` also compiles those files as part of the full suite.
