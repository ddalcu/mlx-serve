# One-shot Windows (NVIDIA CUDA) build from a fresh checkout: checks prerequisites,
# initializes the submodules, stages Zig, MLX + mlx-c and llama.cpp, builds
# zig-out\bin\mlx-serve.exe and packages it with its DLLs in dist\.
#
#   powershell -File scripts\build-windows.ps1
#
# MLX_CUDA_ARCHITECTURES  GPU compute capabilities, e.g. "120" (default "75;86;120": RTX 20-50).
# ZIG_CPU                 CPU baseline for the zig build, e.g. x86_64_v3 (CI); default native.
#
# Building MLX needs the CUDA 13.0 toolkit (CUDA_PATH) and cuDNN 9 (CUDNN_PATH); see
# build-mlx-windows.ps1. The package runs on a machine with an NVIDIA driver, a CUDA 13
# toolkit and cuDNN 9: both engines load those, so no CUDA DLL ships in it.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. (Join-Path $PSScriptRoot "windows-env.ps1")

# MSVC's environment for every step below (fetch-llama.sh reads DLL exports with its dumpbin).
Import-MsvcEnv
$gitBash = Get-GitBash
function Bash($script) {
    & $gitBash -c $script
    if ($LASTEXITCODE) { throw "bash: $script failed" }
}

# Sources and toolchain.
Bash "scripts/check-submodules.sh --fix"
Bash "scripts/fetch-zig.sh"
$zig = Join-Path $Root ".zig-toolchain\zig.exe"

# Engines: MLX + mlx-c (MSVC), llama.cpp (its Windows CUDA release).
& (Join-Path $PSScriptRoot "build-mlx-windows.ps1")
Bash "LLAMA_TARGET=windows scripts/fetch-llama.sh"

# mlx-serve.
$cpu = @(if ($env:ZIG_CPU) { "-Dcpu=$env:ZIG_CPU" }) # @(): an `if` would unroll a one-item array to a string
& $zig build -Dtarget=x86_64-windows-gnu -Doptimize=ReleaseFast @cpu
if ($LASTEXITCODE) { throw "zig build failed" }

# Package: the exe beside every DLL it loads, and MLX's headers for its kernel JIT.
$pkg = Join-Path $Root "dist\mlx-serve-windows-x64"
Remove-Item -Recurse -Force $pkg -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $pkg | Out-Null
Copy-Item zig-out\bin\mlx-serve.exe, lib\mlx-win\bin\*.dll, LICENSE, LICENSE-APACHE-2.0, NOTICE $pkg
Copy-Item -Recurse lib\mlx-win\include "$pkg\include"
$llama = "lib\llama-win\bin"
Copy-Item "$llama\llama.dll", "$llama\ggml.dll", "$llama\ggml-base.dll", "$llama\ggml-cuda.dll", "$llama\ggml-cpu-*.dll", "$llama\libomp.dll" $pkg
Compress-Archive -Force -Path "$pkg\*" -DestinationPath "$pkg.zip"
Write-Host "== built $pkg.zip"
