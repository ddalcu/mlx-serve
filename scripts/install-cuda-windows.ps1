# Install the CUDA toolkit pieces scripts/build-mlx-windows.ps1 builds MLX with, and cuDNN,
# from NVIDIA's redistributable archives (no installer, no driver): CI's Windows runners.
#
#   powershell -File scripts\install-cuda-windows.ps1 -Cuda 13.0.3 -Cudnn 9.27.0 -Dest C:\cuda
#
# Prints the CUDA_PATH and CUDNN_PATH to set (cuDNN lands in <Dest>\cudnn).
param([string]$Cuda = "13.0.3", [string]$Cudnn = "9.27.0", [string]$Dest = "C:\cuda")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$components = "cuda_cccl", "cuda_crt", "cuda_cudart", "cuda_nvcc", "cuda_nvrtc", "cuda_nvtx", "cuda_profiler_api",
    "libcublas", "libcufft", "libcusolver", "libcusparse", "libnvjitlink", "libnvvm", "libnvfatbin", "libnvptxcompiler"
$tmp = Join-Path ([IO.Path]::GetTempPath()) "cuda-redist"
New-Item -ItemType Directory -Force $tmp, $Dest | Out-Null

function Install-Archive($url, $into) {
    $zip = Join-Path $tmp ([IO.Path]::GetFileName($url))
    Write-Host "== $url"
    Invoke-WebRequest -UseBasicParsing $url -OutFile $zip
    tar.exe -xf $zip -C $tmp
    $dir = Join-Path $tmp ([IO.Path]::GetFileNameWithoutExtension($zip))
    robocopy $dir $into /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "copying $dir failed" }
    Remove-Item -Recurse -Force $dir, $zip
}

$base = "https://developer.download.nvidia.com/compute/cuda/redist"
$manifest = Invoke-RestMethod "$base/redistrib_$Cuda.json"
foreach ($c in $components) {
    $win = $manifest.$c."windows-x86_64"
    if (-not $win) { throw "CUDA $Cuda has no windows-x86_64 $c" }
    Install-Archive "$base/$($win.relative_path)" $Dest
}
$cudnnBase = "https://developer.download.nvidia.com/compute/cudnn/redist"
$cudnnWin = (Invoke-RestMethod "$cudnnBase/redistrib_$Cudnn.json").cudnn."windows-x86_64".cuda13
Install-Archive "$cudnnBase/$($cudnnWin.relative_path)" (Join-Path $Dest "cudnn")
Write-Host "CUDA_PATH=$Dest"
Write-Host "CUDNN_PATH=$(Join-Path $Dest 'cudnn')"
