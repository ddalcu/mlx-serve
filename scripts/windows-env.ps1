# Shared by the Windows build scripts (dot-source it).

# MSVC's environment (cl, link, dumpbin), imported from vcvars64.bat unless already present.
function Import-MsvcEnv {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = if (Test-Path $vswhere) { & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath }
    if (-not $vs) { throw "Visual Studio 2022 C++ build tools not found" }
    cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] } }
}

# Git Bash, found beside git.exe: a bare `bash` can be System32's WSL launcher.
function Get-GitBash {
    $git = (Get-Command git -ErrorAction SilentlyContinue).Source
    $bash = if ($git) { Join-Path (Split-Path (Split-Path $git)) "bin\bash.exe" }
    if (-not $bash -or -not (Test-Path $bash)) { throw "Git Bash not found (install Git for Windows)" }
    $bash
}
