# Builds achyudh's llama.cpp ROCm KV-streaming fork for Windows (RDNA 4 / gfx1201 by default).
# Usage: powershell -ExecutionPolicy Bypass -File build.ps1 [-Gpu gfx1201] [-Venv .\.venv-rocm] [-Src .\src]
param(
    [string]$Gpu    = 'gfx1201',
    [string]$Venv   = (Join-Path $PSScriptRoot '.venv-rocm'),
    [string]$Src    = (Join-Path $PSScriptRoot 'src'),
    [string]$Commit = 'b540a4b9b150d06ce9560901dd980a59ec8b49e4'   # pinned upstream commit the patches apply to
)
$ErrorActionPreference = 'Stop'
$Repo = 'https://github.com/achyudh/llama.cpp.git'
$Index = 'https://stable.repo.amd.com/rocm/whl-next/'
function Run { & $args[0] $args[1..($args.Count - 1)]; if ($LASTEXITCODE) { throw "failed: $args" } }

# 1. ROCm SDK in a venv (no system-wide ROCm install on Windows)
if (-not (Test-Path "$Venv\Scripts\python.exe")) {
    Run uv venv --python 3.12 $Venv
    Run uv pip install --python "$Venv\Scripts\python.exe" --index-url $Index `
        "rocm==10.1.0" "rocm-sdk-core==10.1.0" "rocm-sdk-devel==10.1.0" "rocm-sdk-libraries==10.1.0" "rocm-sdk-device-$Gpu==10.1.0"
}
$SP = "$Venv\Lib\site-packages"
$Devel = "$SP\_rocm_sdk_devel"
if (-not (Test-Path "$Devel\lib\llvm\bin\clang.exe")) {
    Run "$Venv\Scripts\rocm-sdk.exe" init      # expands the devel package (~1.3 GB)
}
if (-not (Test-Path "$Devel\lib\llvm\bin\clang.exe")) { throw "ROCm SDK devel not initialised in $Venv" }

# 2. Source at the pinned commit + Windows patches
if (-not (Test-Path "$Src\.git")) {
    Run git init $Src
    Run git -C $Src remote add origin $Repo
}
$head = try { & git -C $Src rev-parse --verify --quiet HEAD 2>$null } catch { $null }   # empty on a fresh repo
if ($head -ne $Commit) {
    Run git -C $Src fetch --depth 1 --progress origin $Commit      # only the pinned commit, no history
    Run git -C $Src checkout --force --detach FETCH_HEAD
}
foreach ($p in Get-ChildItem "$PSScriptRoot\patches\*.patch" | Sort-Object Name) {
    $ErrorActionPreference = 'Continue'      # git writes to stderr when the patch is not applied yet
    & git -C $Src apply --ignore-whitespace --check -R $p.FullName 2>$null
    $reversible = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = 'Stop'
    if ($reversible) { Write-Host "already applied: $($p.Name)"; continue }
    Run git -C $Src apply --ignore-whitespace $p.FullName
    Write-Host "applied: $($p.Name)"
}

# 3. VS Build Tools environment (cmake, ninja, Windows SDK, MSVC runtime libs for lld-link)
$vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vsw -latest -products * -property installationPath
if (-not $vs) { throw "Visual Studio 2022 Build Tools not found (need 'Desktop development with C++')" }
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] }
}
$env:PATH = "$Devel\lib\llvm\bin;$SP\_rocm_sdk_core\bin;$SP\_rocm_sdk_devel\bin;$SP\_rocm_sdk_libraries\bin;$env:PATH"
$env:HIP_PATH = $Devel

# 4. Configure + build
$B = "$Src\build-rocm"
$llvm = "$Devel\lib\llvm\bin"
# CPU code uses a portable AVX2 baseline, not -march=native (a native build crashes with 0xC000001D on other CPUs)
Run cmake -S $Src -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$llvm\clang.exe" "-DCMAKE_CXX_COMPILER=$llvm\clang++.exe" `
    "-DCMAKE_PREFIX_PATH=$Devel\lib\cmake" `
    -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON `
    -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON `
    "-DGPU_TARGETS=$Gpu" "-DAMDGPU_TARGETS=$Gpu" `
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
Run cmake --build $B --target llama-server llama-bench

# 5. HIP DLLs next to the exe, otherwise the older System32 amdhip64 is loaded ("device kernel image is invalid")
foreach ($pat in 'amdhip64_*.dll', 'amd_comgr*.dll', 'hiprtc*.dll') {
    foreach ($d in "$SP\_rocm_sdk_core\bin", "$Devel\bin") {
        Get-ChildItem "$d\$pat" -ErrorAction SilentlyContinue | Copy-Item -Destination "$B\bin" -Force
    }
}

Write-Host "`nBuilt: $B\bin\llama-server.exe" -ForegroundColor Green
