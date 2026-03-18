param(
    [string]$RepoRoot = "C:\codedev\PC_AI",
    [string]$ModelPath = "C:\codedev\PC_AI\Models\Janus-Pro-1B",
    [string]$ImagePath = "C:\codedev\PC_AI\Reports\janus-test-cpu-1b.png",
    [string]$Device = "cuda:auto",
    [int]$Port = 18125,
    [int]$ReadyTimeoutSeconds = 120,
    [string]$Prompt = "Describe this image in one sentence.",
    [uint32]$MaxTokens = 64,
    [double]$Temperature = 0.1
)

$ErrorActionPreference = "Stop"

$artifactExe = Join-Path $RepoRoot ".pcai\build\artifacts\pcai-media\pcai-media.exe"
$buildExe = Join-Path $RepoRoot "Native\pcai_core\target-codex-media-cuda\debug\pcai-media.exe"
$preservedExe = Join-Path $RepoRoot "pcai-media.exe"
$smokeExe = Join-Path $RepoRoot "pcai-media-smoke.exe"
$stdoutLog = Join-Path $RepoRoot "pcai-media-gpu.stdout.log"
$stderrLog = Join-Path $RepoRoot "pcai-media-gpu.stderr.log"

function Resolve-SmokeBinarySource {
    foreach ($candidate in @($artifactExe, $preservedExe, $buildExe)) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    return $null
}

function Get-CudaGpuUsage {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) { return @() }

    $rows = & $nvidiaSmi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader,nounits 2>$null
    foreach ($row in @($rows)) {
        $parts = @($row -split ',')
        if ($parts.Count -lt 5) { continue }
        [PSCustomObject]@{
            Index = [int]($parts[0].Trim())
            Uuid = $parts[1].Trim()
            Name = $parts[2].Trim()
            MemoryUsedMB = [int]($parts[3].Trim())
            MemoryTotalMB = [int]($parts[4].Trim())
        }
    }
}

$binarySource = Resolve-SmokeBinarySource
if (-not $binarySource) {
    throw "No runnable pcai-media binary found in standardized artifacts, repo root, or target-codex-media-cuda."
}

if ($binarySource -ne $preservedExe) {
    Copy-Item $binarySource $preservedExe -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $preservedExe)) {
    throw "Preserved binary not found: $preservedExe"
}

Copy-Item $preservedExe $smokeExe -Force

if (-not (Test-Path $ModelPath)) {
    throw "Model path not found: $ModelPath"
}

if (-not (Test-Path $ImagePath)) {
    throw "Image path not found: $ImagePath"
}

Remove-Item $stdoutLog, $stderrLog -Force -ErrorAction SilentlyContinue
$gpuUsageBefore = @(Get-CudaGpuUsage)

if (-not $env:CUDA_DEVICE_ORDER) {
    $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
}

$process = Start-Process `
    -FilePath $smokeExe `
    -ArgumentList @("--model", $ModelPath, "--device", $Device, "--port", "$Port") `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru `
    -WindowStyle Hidden

try {
    $health = $null
    $readyStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    for ($i = 0; $i -lt $ReadyTimeoutSeconds; $i++) {
        Start-Sleep -Seconds 1
        if ($process.HasExited) {
            throw "Server exited early with code $($process.ExitCode)."
        }

        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
            if ($health.model_loaded -eq $true) {
                $readyStopwatch.Stop()
                break
            }
        }
        catch {
        }
    }

    if ($null -eq $health -or $health.model_loaded -ne $true) {
        throw "Server did not become ready within $ReadyTimeoutSeconds seconds."
    }

    $imageBytes = [System.IO.File]::ReadAllBytes($ImagePath)
    $body = @{
        image_base64 = [Convert]::ToBase64String($imageBytes)
        prompt       = $Prompt
        max_tokens   = $MaxTokens
        temperature  = $Temperature
    } | ConvertTo-Json -Depth 4

    $understandStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $understand = Invoke-RestMethod `
        -Uri "http://127.0.0.1:$Port/v1/images/understand" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 180
    $understandStopwatch.Stop()
    $gpuUsageAfter = @(Get-CudaGpuUsage)
    $gpuUsageDelta = foreach ($afterGpu in $gpuUsageAfter) {
        $beforeGpu = @($gpuUsageBefore | Where-Object Index -eq $afterGpu.Index | Select-Object -First 1)
        [PSCustomObject]@{
            Index = $afterGpu.Index
            Uuid = $afterGpu.Uuid
            Name = $afterGpu.Name
            MemoryDeltaMB = if ($beforeGpu.Count -gt 0) { $afterGpu.MemoryUsedMB - $beforeGpu[0].MemoryUsedMB } else { $null }
            MemoryUsedBeforeMB = if ($beforeGpu.Count -gt 0) { $beforeGpu[0].MemoryUsedMB } else { $null }
            MemoryUsedAfterMB = $afterGpu.MemoryUsedMB
            MemoryTotalMB = $afterGpu.MemoryTotalMB
        }
    }
    $selectedGpu = @($gpuUsageDelta | Sort-Object @{ Expression = 'MemoryDeltaMB'; Descending = $true }, @{ Expression = 'Index'; Ascending = $true } | Select-Object -First 1)

    [pscustomobject]@{
        health = $health
        understand = $understand
        metrics = [pscustomobject]@{
            device = $Device
            cudaDeviceOrder = $env:CUDA_DEVICE_ORDER
            cudaVisibleDevices = $env:CUDA_VISIBLE_DEVICES
            readyMs = [math]::Round($readyStopwatch.Elapsed.TotalMilliseconds, 1)
            understandMs = [math]::Round($understandStopwatch.Elapsed.TotalMilliseconds, 1)
            outputChars = if ($understand.text) { $understand.text.Length } else { 0 }
            selectedGpu = if ($selectedGpu.Count -gt 0) { $selectedGpu[0] } else { $null }
            gpuUsageDelta = @($gpuUsageDelta)
            approxTokensPerSec = if ($understand.text) {
                $tokenCount = ([regex]::Matches($understand.text, '\S+')).Count
                if ($understandStopwatch.Elapsed.TotalSeconds -gt 0) {
                    [math]::Round($tokenCount / $understandStopwatch.Elapsed.TotalSeconds, 2)
                } else {
                    0
                }
            } else {
                0
            }
        }
        binarySource = Get-Item $binarySource | Select-Object FullName, Length, LastWriteTime
        preservedExe = Get-Item $preservedExe | Select-Object FullName, Length, LastWriteTime
        smokeExe = Get-Item $smokeExe | Select-Object FullName, Length, LastWriteTime
        stderrLog = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 120 } else { @() }
        stdoutLog = if (Test-Path $stdoutLog) { Get-Content $stdoutLog -Tail 80 } else { @() }
    } | ConvertTo-Json -Depth 6
}
finally {
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
}
