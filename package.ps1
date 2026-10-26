# Packages the build output + the ROCm runtime pieces it needs into dist\*.zip, so users need only the AMD driver.
# Run after build.ps1.  Usage: powershell -ExecutionPolicy Bypass -File package.ps1 [-Gpu gfx1201]
param(
    [string]$Gpu  = 'gfx1201',
    [string]$Venv = (Join-Path $PSScriptRoot '.venv-rocm'),
    [string]$Src  = (Join-Path $PSScriptRoot 'src')
)
$ErrorActionPreference = 'Stop'
$SP   = "$Venv\Lib\site-packages"
$Bin  = "$Src\build-rocm\bin"
$Libs = "$SP\_rocm_sdk_libraries\bin"
$sha  = (& git -C $Src rev-parse --short HEAD).Trim()
$name = "llama-kvstream-rocm-windows-$Gpu-$sha"
$Out  = Join-Path $PSScriptRoot "dist\$name"

if (-not (Test-Path "$Bin\llama-server.exe")) { throw "Build first: $Bin\llama-server.exe not found" }
Remove-Item $Out -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory $Out -Force | Out-Null

# llama.cpp binaries + HIP runtime DLLs (build.ps1 already copied amdhip64/amd_comgr/hiprtc next to the exe)
Copy-Item "$Bin\*.exe", "$Bin\*.dll" $Out
# BLAS stack that ggml-hip.dll pulls in: hipblas -> rocsolver, rocblas -> libhipblaslt, plus their GPU kernel files
foreach ($f in 'hipblas.dll', 'rocsolver.dll', 'rocblas.dll', 'libhipblaslt.dll', 'origami.dll') { Copy-Item "$Libs\$f" $Out }
Copy-Item "$SP\_rocm_sdk_core\bin\rocm_kpack.dll" $Out      # loaded by amdhip64_7.dll

# Fail the package if any DLL still has an import that is neither in the folder nor in System32
$llvmObjdump = "$SP\_rocm_sdk_devel\lib\llvm\bin\llvm-objdump.exe"
$missing = foreach ($f in Get-ChildItem $Out -File | Where-Object { $_.Extension -in '.dll', '.exe' }) {
    & $llvmObjdump -p $f.FullName | Select-String 'DLL Name' | ForEach-Object {
        $dep = ($_.Line -replace '.*DLL Name:\s*', '').Trim()
        if ($dep -notmatch '^(api-ms|ext-ms)' -and -not (Test-Path "$Out\$dep") -and -not (Test-Path "$env:SystemRoot\System32\$dep")) { "$dep (needed by $($f.Name))" }
    }
}
if ($missing) { throw "Unresolved DLL imports:`n$($missing | Sort-Object -Unique | Out-String)" }
foreach ($d in 'rocblas', 'hipblaslt') { Copy-Item "$Libs\$d" $Out -Recurse }

Copy-Item "$PSScriptRoot\run-example.ps1", "$PSScriptRoot\LICENSE" $Out
Copy-Item "$SP\_rocm_sdk_core\share\doc\amd_comgr\LICENSE.txt" "$Out\LICENSE-amd_comgr.txt"
@"
llama.cpp ROCm KV-streaming fork, Windows build ($Gpu)
Source: https://github.com/achyudh/llama.cpp (branch feature/adaptive-kv-stream-rocm, commit $sha) + patches/0001-windows-hip-no-hsa.patch
Includes unmodified AMD ROCm 10.1.0 runtime libraries (HIP, comgr, hiprtc, rocBLAS, hipBLASLt, hipBLAS, rocSOLVER) from AMD's pip wheels:
https://stable.repo.amd.com/rocm/whl-next/ ; their sources and licenses: https://github.com/ROCm
llama.cpp itself is MIT licensed (LICENSE).

Needs: AMD Adrenalin driver for your GPU, Microsoft Visual C++ 2015-2022 Redistributable (x64).
Run:   powershell -ExecutionPolicy Bypass -File run-example.ps1 -Model C:\models\your-model.gguf -Device ROCm0
       (with an iGPU present the dGPU is usually ROCm1; check the startup log)
"@ | Set-Content "$Out\README.txt" -Encoding utf8

$zip = "$Out.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path "$Out\*" -DestinationPath $zip -CompressionLevel Optimal
"{0}: {1:N0} MB zipped ({2:N0} MB unpacked)" -f $zip, ((Get-Item $zip).Length / 1MB), ((Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
