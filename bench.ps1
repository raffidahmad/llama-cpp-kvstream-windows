# Quick, comparable benchmark against a RUNNING llama-server (start it with run-example.ps1 first).
# Prints a markdown block (system info + results) to paste into a "Test results" issue.
# Usage: powershell -ExecutionPolicy Bypass -File bench.ps1 [-Port 8080] [-ApiKey xxx]
param(
    [int]$Port = 8080,
    [string]$ApiKey
)
$ErrorActionPreference = 'Stop'
$base = "http://127.0.0.1:$Port"
$hdr = @('-H', 'Content-Type: application/json')
if ($ApiKey) { $hdr += '-H', "Authorization: Bearer $ApiKey" }

function Ask($prompt, $maxTokens, $ignoreEos) {
    $body = @{
        messages = @(@{ role = 'user'; content = $prompt }); max_tokens = $maxTokens; temperature = 0
        ignore_eos = $ignoreEos; chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 5 -Compress
    $f = [IO.Path]::GetTempFileName(); Set-Content $f $body -Encoding ascii
    $out = & curl.exe -s --max-time 600 @hdr -d "@$f" "$base/v1/chat/completions"
    Remove-Item $f
    try { ($out | ConvertFrom-Json).timings } catch { throw "No valid response from $base : $out" }
}

try { $props = & curl.exe -s @hdr "$base/props" | ConvertFrom-Json } catch { $props = $null }
$gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Radeon|AMD' } | ForEach-Object { "$($_.Name) (driver $($_.DriverVersion))" }) -join '; '
$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$os = (Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).BuildNumber

Write-Host 'Warm-up...' -ForegroundColor DarkGray
$null = Ask 'Say hello.' 16 $false

# Three DIFFERENT code prompts: the same prompt twice makes the n-gram drafter copy its own earlier answer and inflates t/s.
$codePrompts = @(
    'Write a complete Python implementation of an LRU cache class with get, put, and unit tests. Code only.',
    'Write a Python function that parses an ISO-8601 date string without any libraries, with unit tests. Code only.',
    'Write a Python implementation of a thread-safe bounded queue with put, get, and unit tests. Code only.'
)
$tg = @()
foreach ($p in $codePrompts) {
    Write-Host "Code prompt $($tg.Count + 1)/3..." -ForegroundColor DarkGray
    $t = Ask $p 512 $true; $tg += [math]::Round($t.predicted_per_second, 1)
}

# Long prompts of non-repeating lines (~9k tokens) to measure prefill and decode. The first one is a throwaway:
# the first long prefill after a server start is often much slower than later ones.
function LongPrompt($seed) {
    $rng = [Random]::new($seed)
    $lines = 1..450 | ForEach-Object { "Record ${_}: " + (-join (1..12 | ForEach-Object { [char]$rng.Next(97, 123) })) + " value=" + $rng.Next(1000, 999999) }
    ($lines -join "`n") + "`n`nHow many records are listed above? Answer in one sentence."
}
Write-Host 'Long prompt (warm-up)...' -ForegroundColor DarkGray
$null = Ask (LongPrompt 1) 32 $true
Write-Host 'Long prompt...' -ForegroundColor DarkGray
$t = Ask (LongPrompt 42) 128 $true

$serverInfo = 'n/a'
if ($props) {
    $mdl = if ($props.model_alias) { $props.model_alias } else { $props.default_generation_settings.model }
    $serverInfo = "model=$mdl, n_ctx=$($props.default_generation_settings.n_ctx)"
}
$sorted = $tg | Sort-Object
@"

<!-- paste everything below into the issue -->
**System:** $gpu | $ramGB GB RAM | $os
**Server:** $serverInfo

| Test | Result |
|---|---|
| Code decode, 3 prompts x 512 tokens (t/s) | $($tg -join ' / ') (median $($sorted[1])) |
| Long prompt: $($t.prompt_n) tokens prefill (t/s) | $([math]::Round($t.prompt_per_second, 1)) |
| Decode after long prompt (t/s) | $([math]::Round($t.predicted_per_second, 1)) |
"@
