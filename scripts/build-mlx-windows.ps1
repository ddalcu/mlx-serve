# Build mlx (CUDA) + mlx-c as MSVC DLLs from the pinned submodules and stage them in
# lib/mlx-win for the Windows mlx-serve.exe: include/ (headers, CCCL; ships beside mlx.dll
# for MLX's kernel JIT), lib/ (mlxc.lib), bin/ (mlx.dll, mlxc.dll, libopenblas.dll).
# The Windows counterpart of scripts/build-mlx-linux.sh.
#
# Needs: Visual Studio 2022 C++ build tools, CMake, Ninja, Git, the CUDA 13.0 toolkit
# (CUDA_PATH; 13.3's nvcc cannot compile MLX's CUTLASS kernels with MSVC, and MLX's own
# Windows CI builds with 13.0) and cuDNN 9 for CUDA 13 (CUDNN_PATH).
# Users run the result against any CUDA 13.x runtime (see patches/mlx-cuda-windows-toolkit-dirs.patch).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Mlx = Join-Path $Root "lib\mlx-src"
$Mlxc = Join-Path $Root "lib\mlxc-src"
$Stage = Join-Path $Root "lib\mlx-win"
$Work = Join-Path $Root ".work-mlx-windows"
$Arch = if ($env:MLX_CUDA_ARCHITECTURES) { $env:MLX_CUDA_ARCHITECTURES } else { "75;86;120" }

foreach ($dir in $Mlx, $Mlxc) {
    if (-not (Test-Path (Join-Path $dir "CMakeLists.txt"))) { throw "$dir is not checked out: git submodule update --init lib/mlx-src lib/mlxc-src" }
}
# Idempotent: the stamp names every input, so a matching stage (a CI cache hit) skips the build.
$patches = Get-ChildItem (Join-Path $Root "patches") -Filter "mlx*.patch" | Sort-Object Name
$bytes = [byte[]]$(foreach ($f in @($patches) + (Get-Item $PSCommandPath)) { [IO.File]::ReadAllBytes($f.FullName) })
$hash = (Get-FileHash -InputStream ([IO.MemoryStream]::new($bytes))).Hash.Substring(0, 12)
$want = "mlx=$(git -C $Mlx rev-parse HEAD) mlxc=$(git -C $Mlxc rev-parse HEAD) cuda_arch=$Arch inputs=$hash"
$stamp = Join-Path $Stage ".version"
if ((Test-Path (Join-Path $Stage "bin\mlx.dll")) -and (Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $want)) {
    Write-Host "== lib/mlx-win already staged ($want) - nothing to do"; exit 0
}

foreach ($c in "cmake", "ninja") { if (-not (Get-Command $c -ErrorAction SilentlyContinue)) { throw "$c not found (install CMake and Ninja)" } }
if (-not $env:CUDA_PATH) { throw "CUDA_PATH is not set (install the CUDA 13.0 toolkit)" }
if (-not $env:CUDNN_PATH) { throw "CUDNN_PATH is not set (install cuDNN 9 for CUDA 13)" }
$nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
if ((& $nvcc --version | Out-String) -notmatch "release 13\.0") { throw "$nvcc is not CUDA 13.0" }
$cudnnInc = Join-Path $env:CUDNN_PATH "include"
$cudnnLib = Get-ChildItem (Join-Path $env:CUDNN_PATH "lib") -Recurse -Filter cudnn.lib | Select-Object -First 1
if (-not $cudnnLib) { throw "no cudnn.lib under $env:CUDNN_PATH\lib" }

. (Join-Path $PSScriptRoot "windows-env.ps1")
Import-MsvcEnv
$env:Path = "$env:CUDA_PATH\bin;$env:Path"

# Each patch once: applied when it applies, skipped when it is already in (it reverses).
# --ignore-whitespace: Git for Windows checks the sources out with CRLF line endings. Tests and
# docs are not built, and two patches add tests at the same spot, which no single one reverses past.
# The checks fail by design, so their stderr must not stop the script (Windows PowerShell 5.1
# turns redirected native stderr into errors).
function Apply-Patch($repo, $patch) {
    $ErrorActionPreference = "Continue"
    $opts = "--ignore-whitespace", "--exclude=tests/*", "--exclude=docs/*"
    git -C $repo apply @opts --check $patch 2>$null
    if ($LASTEXITCODE -eq 0) { git -C $repo apply @opts $patch; Write-Host "   applied $(Split-Path -Leaf $patch)"; return }
    git -C $repo apply @opts --reverse --check $patch 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "   already in: $(Split-Path -Leaf $patch)"; return }
    throw "patch $patch does not apply to $repo"
}
Write-Host "== patch mlx + mlx-c"
foreach ($p in $patches) {
    Apply-Patch $(if ($p.Name -like "mlxc-*") { $Mlxc } else { $Mlx }) $p.FullName
}

$prefix = Join-Path $Work "prefix"
Write-Host "== build mlx (CUDA $Arch)"
cmake -S $Mlx -B (Join-Path $Work "mlx-build") -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON `
    -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON -DMLX_BUILD_CUDA=ON "-DMLX_CUDA_ARCHITECTURES=$Arch" `
    "-DCMAKE_CUDA_COMPILER=$($nvcc -replace '\\','/')" "-DCUDAToolkit_ROOT=$($env:CUDA_PATH -replace '\\','/')" `
    "-DCUDNN_INCLUDE_PATH=$($cudnnInc -replace '\\','/')" "-DCUDNN_LIBRARY_PATH=$($cudnnLib.DirectoryName -replace '\\','/')" `
    -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_GGUF=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_EXAMPLES=OFF
if ($LASTEXITCODE) { throw "mlx configure failed" }
cmake --build (Join-Path $Work "mlx-build") --parallel
if ($LASTEXITCODE) { throw "mlx build failed" }
cmake --install (Join-Path $Work "mlx-build") --prefix $prefix | Out-Null
if ($LASTEXITCODE) { throw "mlx install failed" }

Write-Host "== build mlx-c"
cmake -S $Mlxc -B (Join-Path $Work "mlxc-build") -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON `
    -DMLX_C_USE_SYSTEM_MLX=ON -DMLX_C_BUILD_EXAMPLES=OFF "-DCMAKE_PREFIX_PATH=$prefix" `
    "-DCUDAToolkit_ROOT=$($env:CUDA_PATH -replace '\\','/')" "-DCUDNN_INCLUDE_PATH=$($cudnnInc -replace '\\','/')" "-DCUDNN_LIBRARY_PATH=$($cudnnLib.DirectoryName -replace '\\','/')"
if ($LASTEXITCODE) { throw "mlx-c configure failed" }
cmake --build (Join-Path $Work "mlxc-build") --target mlxc --parallel
if ($LASTEXITCODE) { throw "mlx-c build failed" }
cmake --install (Join-Path $Work "mlxc-build") --prefix $prefix | Out-Null
if ($LASTEXITCODE) { throw "mlx-c install failed" }

Write-Host "== stage $Stage"
Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$Stage\lib", "$Stage\bin" | Out-Null
Copy-Item -Recurse (Join-Path $prefix "include") "$Stage\include"
Copy-Item (Join-Path $prefix "lib\mlxc.lib") "$Stage\lib\"
Copy-Item (Get-ChildItem $prefix -Recurse -Include mlx.dll, mlxc.dll, libopenblas.dll).FullName "$Stage\bin\"
Set-Content $stamp $want
Write-Host "== done: $Stage ($want)"
