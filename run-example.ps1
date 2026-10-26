# Minimal launcher: sets the ROCm runtime env + HIP queue settings, then starts llama-server with KV streaming.
# Works from the release zip (DLLs next to the exe) and from a repo build (runtime from .venv-rocm).
# Usage: powershell -ExecutionPolicy Bypass -File run-example.ps1 -Model C:\models\model.gguf [-Ctx 262144] [-Port 8080]
param(
    [Parameter(Mandatory)][string]$Model,
    [int]$Ctx = 262144,
    [int]$Port = 8080,
    [string]$Venv,                      # repo build: ROCm venv (default .\.venv-rocm)
    [string]$Bin,                       # folder with llama-server.exe (default: this folder, else the repo build)
    [string]$Device = 'ROCm0',          # check the startup log; with an iGPU the dGPU is often ROCm1
    [int]$ResidentMiB = 300,            # --kv-resident-mib (prefill); lower if you run out of VRAM
    [int]$ResidentDecodeMiB = 1800,     # --kv-resident-mib-decode
    [string]$ApiKey,                    # optional
    [switch]$NoSpec                     # disable MTP + n-gram speculation (use for models without an MTP head)
)
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Bin) { $Bin = if (Test-Path "$Root\llama-server.exe") { $Root } else { "$Root\src\build-rocm\bin" } }
if (-not $Venv) { $Venv = "$Root\.venv-rocm" }
if (Test-Path "$Venv\Lib\site-packages") {          # repo build: runtime comes from the venv
    $SP = "$Venv\Lib\site-packages"
    $env:PATH = "$SP\_rocm_sdk_core\bin;$SP\_rocm_sdk_devel\bin;$SP\_rocm_sdk_libraries\bin;$env:PATH"
    $env:HIP_PATH = "$SP\_rocm_sdk_devel"
} else {                                            # release zip: everything sits next to the exe
    $env:PATH = "$Bin;$env:PATH"
}
$env:GPU_MAX_HW_QUEUES = '2'
$env:DEBUG_HIP_FORCE_ASYNC_QUEUE = '1'
$env:DEBUG_HIP_FORCE_GRAPH_QUEUES = '1'

$srvArgs = @(
    '--model', $Model, '--host', '127.0.0.1', '--port', $Port,
    '--ctx-size', $Ctx, '--device', $Device, '-fit', 'off', '--n-gpu-layers', 'all',
    '--flash-attn', 'on', '--jinja', '--parallel', 1,
    '--cache-type-k', 'q8_0', '--cache-type-v', 'q4_0',
    '--kv-stream', '--kv-resident-mib', $ResidentMiB, '--kv-resident-mib-decode', $ResidentDecodeMiB,
    '--batch-size', 2048, '--ubatch-size', 512
)
if (-not $NoSpec) {     # the speculative settings are the biggest speed lever (see README Results)
    $srvArgs += '--cache-type-k-draft', 'q8_0', '--cache-type-v-draft', 'q4_0',
        '--spec-type', 'draft-mtp,ngram-mod', '--spec-rs-replay', '--spec-draft-n-max', 4,
        '--spec-ngram-mod-n-max', 32, '--spec-ngram-mod-n-min', 12, '--spec-ngram-mod-n-match', 24,
        '--spec-draft-sampling', 'probabilistic'
}
if ($ApiKey) { $srvArgs += '--api-key', $ApiKey }
& "$Bin\llama-server.exe" @srvArgs
