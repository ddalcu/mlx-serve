[English](../building.md) · [简体中文](building.md)

# 从源码构建

只有在你动手改 mlx-serve 时才需要这些内容。只是想用它，就获取[应用](https://github.com/ddalcu/mlx-serve/releases/latest) 或执行 `brew install mlx-serve`。

## 前置条件

- macOS 26.2+ 与 Apple Silicon（M1/M2/M3/M4/M5）—— 内置的 MLX 以 deployment target 26.2 构建，因此 M5 神经加速器（NAX）kernel 是启用状态随包的
- Xcode 26.2+ 与 Metal Toolchain 组件 —— mlx + mlx-c 是由 `scripts/build-mlx.sh` 编译的固定子模块，不是 brew 包，所以 brew bottle 静默省略的那些 NAX kernel 都被包含进来。Xcode 26 把 Metal 编译器作为单独的下载项发布，因此如果 `xcrun -sdk macosx metal --version` 失败，先运行 `xcodebuild -downloadComponent MetalToolchain`
- cmake 与 libwebp：在仓库根目录执行 `brew bundle install --file=Brewfile`。cmake 构建 mlx 子模块，webp 在视觉流水线中解码图像（`webp >= 1.6.0`，构建时检查）
- [Zig 0.17](https://ziglang.org/download/) —— 由 `./scripts/fetch-zig.sh` 自动落到 `.zig-toolchain/`

## 应用 + 服务器

一个脚本构建全部：

```bash
git clone --recurse-submodules https://github.com/ddalcu/mlx-serve && cd mlx-serve
brew bundle install --file=Brewfile
./app/build.sh
open "app/MLX-Serve.app"
```

`app/build.sh` 会把固定子模块拉回各自的 commit，落位 llama.cpp 与 Zig，在断言 NAX kernel 的前提下构建 mlx + mlx-c，编译 Swift 应用与 Zig 服务器，然后打包并签名。环境里没有签名身份时，它做 ad-hoc 签名并跳过公证，因此不需要 Apple 开发者账号。版本由 `.github/workflows/release.yml` 发布。

## 只构建服务器

```bash
./scripts/fetch-zig.sh                               # 把固定的 Zig 落位到 .zig-toolchain/
export PATH="$PWD/.zig-toolchain:$PATH"
./scripts/fetch-llama.sh && ./scripts/build-mlx.sh   # 执行一次，之后每次更新 pin 时再执行
zig build -Doptimize=ReleaseFast                     # 始终用 ReleaseFast；Debug 慢 2-4 倍
```

## Linux 与 Windows（NVIDIA）

每个版本都附带 `mlx-serve-linux-x64-cuda13.tar.gz`（RTX 20 到 50 系列）和 `mlx-serve-linux-arm64-cuda13.tar.gz`（GH200、Jetson Thor、DGX Spark）。它们需要 580 或更新的 NVIDIA 驱动、CUDA 13 工具包、cuDNN 9、OpenBLAS/LAPACKE、libwebp，以及 Avahi 的 dns_sd 兼容库。解压到一个空文件夹（压缩包里没有顶层目录），然后运行 `zig-out/bin/mlx-serve`；它会在自己旁边找到所需的库。

在 Linux 上构建：从全新检出的仓库运行 `./scripts/build-linux.sh`。在 Windows 上，`powershell -File scripts\build-windows.ps1` 会构建 `mlx-serve.exe` 并打包成 `dist\mlx-serve-windows-x64.zip`；构建需要 Visual Studio 2022 C++ 工具、CMake、Ninja、Git、CUDA 13.0 工具包与 cuDNN 9，运行则需要带 CUDA 13 与 cuDNN 9 的 NVIDIA 驱动。kernel、各设备默认值与内存说明见 [CUDA.md](../CUDA.md)（英文）。

## 封闭式测试（Linux 上也可以）

逐步预览编码器（`src/sdk/media/preview.zig` + `src/sdk/media/jpeg.zig` + `src/sdk/media/latent_rgb.zig`）不链接 MLX，也不链接 Homebrew webp，因此任何机器（包括 Linux Cloud Agent）都能在没有 GPU 的情况下构建并运行它：

```bash
./scripts/fetch-zig.sh
export PATH="$PWD/.zig-toolchain:$PATH"
zig build preview-test
```

这个 step 在 Mac 上也存在，并构建同样的封闭式产物，但它**不是**在未落位 mlx 的情况下构建的办法：`verifyBrewDeps` 与 `verifyMlxStage` 在配置阶段对每个 step 都会运行，所以 `lib/mlx/` 必须已经构建好。在 Mac 上，`zig build test` 也会把这些文件作为完整测试套件的一部分一起编译。
