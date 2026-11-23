#Requires -Version 7.0
<#
.SYNOPSIS
Measure prefill and decode on a specifically selected llama-bench device.
.DESCRIPTION
Uses a staged Windows llama.cpp build and one local GGUF file. Verify the
distribution, including its runtime DLLs, before staging it. This command
checks the executable and model digests, records the requested and observed
device, and emits one report object after both benchmark slices pass.
The report includes the local backend sidecar digest when that file exists.
MmqPrecision requests a CUDA setting. The q4 route requires a matching device
capability of 10.0 or newer. Placement evidence and timings do not identify the
instructions executed by a kernel.
.EXAMPLE
./Invoke-PcaiLlamaDeviceBenchmark.ps1 -BenchPath $bench -BenchSha256 $benchHash -ModelPath $model -ModelSha256 $modelHash -Route IntelVulkan
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BenchPath,
    [Parameter(Mandatory)][ValidatePattern('\A[0-9a-fA-F]{64}\z')][string]$BenchSha256,
    [Parameter(Mandatory)][string]$ModelPath,
    [Parameter(Mandatory)][ValidatePattern('\A[0-9a-fA-F]{64}\z')][string]$ModelSha256,
    [Parameter(Mandatory)][ValidateSet('CPU', 'IntelVulkan', 'NvidiaCuda')][string]$Route,
    [ValidateSet('q4', 'q8')][string]$MmqPrecision,
    [ValidateRange(1, 2048)][int]$PromptTokens = 256,
    [ValidateRange(1, 256)][int]$DecodeTokens = 64,
    [ValidateRange(1, 10)][int]$Repetitions = 3,
    [ValidateRange(1, 32)][int]$Threads = 4,
    [ValidateRange(1, 180)][int]$TimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'

function Invoke-LlamaBenchmarkChild {
    param(
        [string]$FilePath,
        [string]$WorkingDirectory,
        [string[]]$Arguments,
        [AllowNull()][string]$Precision,
        [int]$TimeoutSeconds
    )

    # The job-managed launcher reads data from stdin and applies the override
    # to its child only. Forward both streams without PowerShell text handling.
    $launcher = @'
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false, $true)
$request = ConvertFrom-Json -InputObject ([Console]::In.ReadToEnd()) -AsHashtable
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $request.filePath
$start.WorkingDirectory = $request.workingDirectory
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
foreach ($argument in $request.arguments) {
    $start.ArgumentList.Add($argument)
}
if ([string]::IsNullOrEmpty($request.precision)) {
    [void]$start.Environment.Remove('GGML_CUDA_MMQ_PREC')
} else {
    $start.Environment['GGML_CUDA_MMQ_PREC'] = $request.precision
}
$child = [Diagnostics.Process]::Start($start)
try {
    $child.StandardInput.Close()
    $stdout = $child.StandardOutput.BaseStream.CopyToAsync([Console]::OpenStandardOutput())
    $stderr = $child.StandardError.BaseStream.CopyToAsync([Console]::OpenStandardError())
    $child.WaitForExit()
    [Threading.Tasks.Task]::WhenAll([Threading.Tasks.Task[]]@($stdout, $stderr)).GetAwaiter().GetResult()
    exit $child.ExitCode
} finally {
    $child.Dispose()
}
'@
    $inputText = ConvertTo-Json -Compress -InputObject @{
        filePath = $FilePath
        workingDirectory = $WorkingDirectory
        arguments = $Arguments
        precision = $Precision
    }
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($launcher))
    $pwshName = if ($IsWindows) { 'pwsh.exe' } else { 'pwsh' }
    $launcherArguments = @('-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', $encoded)
    Invoke-VigilBoundedProcess -FilePath (Join-Path $PSHOME $pwshName) -WorkingDirectory $WorkingDirectory `
        -Arguments $launcherArguments -InputText $inputText -TimeoutSeconds $TimeoutSeconds
}

if ($MmqPrecision -and $Route -ne 'NvidiaCuda') {
    throw 'MmqPrecision is supported only for NvidiaCuda.'
}

$bench = Get-Item -LiteralPath $BenchPath
$model = Get-Item -LiteralPath $ModelPath
if ($bench -isnot [IO.FileInfo] -or $bench.Extension -ne '.exe') {
    throw 'BenchPath must identify a staged Windows executable.'
}
if ($model -isnot [IO.FileInfo] -or $model.Extension -ne '.gguf' -or
    $model.FullName.Contains(',') -or $model.Name -match '-[0-9]{5}-of-[0-9]{5}\.gguf$') {
    throw 'ModelPath must identify one local GGUF file without a comma or split-file suffix.'
}
foreach ($item in @(
        @{ Path = $bench.FullName; Expected = $BenchSha256 },
        @{ Path = $model.FullName; Expected = $ModelSha256 }
    )) {
    if ((Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash -ne $item.Expected) {
        throw "SHA256 mismatch for '$($item.Path)'."
    }
}

. (Join-Path $PSScriptRoot '../../Tools/SystemScripts/Networking/Invoke-VigilBoundedProcess.ps1')

$backendDllName = switch ($Route) {
    'IntelVulkan' { 'ggml-vulkan.dll' }
    'NvidiaCuda' { 'ggml-cuda.dll' }
}
$backendSidecar = if ($backendDllName) {
    $backendDllPath = Join-Path $bench.DirectoryName $backendDllName
    if (Test-Path -LiteralPath $backendDllPath -PathType Leaf) {
        [pscustomobject]@{
            FileName = $backendDllName
            Sha256 = (Get-FileHash -LiteralPath $backendDllPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
}

$device = 'none'
$description = $null
$cudaCapability = $null
$gpuLayers = 0
$childPrecision = if ($MmqPrecision) { $MmqPrecision.ToLowerInvariant() } else { $null }
if ($Route -ne 'CPU') {
    $probe = Invoke-LlamaBenchmarkChild -FilePath $bench.FullName -WorkingDirectory $bench.DirectoryName -Arguments @('--list-devices') -Precision $childPrecision -TimeoutSeconds 10
    if ($probe.ExitCode -ne 0) {
        throw "llama-bench device discovery exited with code $($probe.ExitCode): $($probe.Stderr)"
    }
    $labelPattern = if ($Route -eq 'IntelVulkan') {
        '(?<id>Vulkan[0-9]+): (?<name>Intel\b[^\r\n]+?)'
    } else {
        '(?<id>CUDA[0-9]+): (?<name>NVIDIA\b[^\r\n]+?)'
    }
    $devices = [regex]::Matches($probe.Stdout, "(?im)^[ \t]+$labelPattern \([0-9]+ MiB, [0-9]+ MiB free\)\r?$")
    if ($devices.Count -ne 1) {
        throw "Expected exactly one current llama-bench device for $Route, found $($devices.Count)."
    }
    $device = $devices[0].Groups['id'].Value
    $description = $devices[0].Groups['name'].Value
    if ($Route -eq 'NvidiaCuda') {
        $index = $device.Substring(4)
        $escapedName = [regex]::Escape($description)
        $reported = [regex]::Matches($probe.Stderr, "(?im)^[ \t]*Device ${index}: $escapedName, compute capability (?<cap>[0-9]+\.[0-9]+)(?:,|\s|$)")
        $caps = @($reported | ForEach-Object { $_.Groups['cap'].Value } | Select-Object -Unique)
        if ($caps.Count -eq 1) { $cudaCapability = [version]$caps[0] }
        if ($MmqPrecision -eq 'q4' -and (!$cudaCapability -or $cudaCapability.Major -lt 10)) {
            throw "A q4 MMQ comparison requires confirmed CUDA compute capability 10.0 or newer for $device."
        }
    }
    $gpuLayers = 999
}

$arguments = @('--offline', '-m', $model.FullName, '-p', "$PromptTokens", '-n', "$DecodeTokens",
    '-r', "$Repetitions", '-t', "$Threads", '-dev', $device, '-ngl', "$gpuLayers", '-o', 'json', '-v')
$run = Invoke-LlamaBenchmarkChild -FilePath $bench.FullName -WorkingDirectory $bench.DirectoryName -Arguments $arguments -Precision $childPrecision -TimeoutSeconds $TimeoutSeconds

if ($run.ExitCode -ne 0) {
    throw "llama-bench exited with code $($run.ExitCode): $($run.Stderr)"
}
$rows = ConvertFrom-Json -InputObject $run.Stdout -NoEnumerate -AsHashtable
if ($rows -isnot [array] -or $rows.Count -ne 2) {
    throw 'Expected exactly two benchmark rows for prefill and decode.'
}

$measurements = [ordered]@{}
foreach ($row in $rows) {
    if ($row -isnot [Collections.IDictionary] -or $row.devices -cne $device -or
        $row.model_filename -ne $model.FullName -or $row.n_gpu_layers -ne $gpuLayers -or
        $row.n_threads -ne $Threads -or $row.build_number -isnot [ValueType] -or
        $row.build_number -le 0 -or $row.build_number -ne $rows[0].build_number -or
        $row.build_commit -cne $rows[0].build_commit -or $row.build_commit -notmatch '\A[0-9a-fA-F]{7,40}\z') {
        throw 'A benchmark row disagrees with the selected model, device or build.'
    }
    $slice = if ($row.n_prompt -eq $PromptTokens -and $row.n_gen -eq 0) { 'Prefill' }
    elseif ($row.n_prompt -eq 0 -and $row.n_gen -eq $DecodeTokens) { 'Decode' }
    if (-not $slice -or $measurements.Contains($slice) -or
        $row.samples_ts -isnot [array] -or $row.samples_ns -isnot [array] -or
        $row.samples_ts.Count -ne $Repetitions -or $row.samples_ns.Count -ne $Repetitions -or
        $row.avg_ts -isnot [ValueType] -or -not [double]::IsFinite([double]$row.avg_ts) -or $row.avg_ts -le 0) {
        throw 'Expected one prefill and one decode slice with the requested number of positive measurements.'
    }
    foreach ($sample in ($row.samples_ts + $row.samples_ns)) {
        if ($sample -isnot [ValueType] -or -not [double]::IsFinite([double]$sample) -or $sample -le 0) {
            throw 'Benchmark samples must be positive finite numbers.'
        }
    }
    $measurements[$slice] = [pscustomobject]@{
        TokensPerSecond = [double]$row.avg_ts
        SamplesTokensPerSecond = [double[]]$row.samples_ts
        SamplesNanoseconds = [long[]]$row.samples_ns
    }
}

$selected = [regex]::Matches($run.Stderr, '(?im)\busing device (?<id>[A-Za-z][A-Za-z0-9]*[0-9]) \((?<name>[^\r\n]+)\) \(')
$offloads = [regex]::Matches($run.Stderr, '(?im)\boffloaded (?<used>[0-9]+)/(?<total>[0-9]+) layers to GPU\b')
$offloadedLayers = 0
$totalLayers = $null
$modelBufferMiB = $null
if ($Route -eq 'CPU') {
    if ($selected.Count -or @($offloads | Where-Object { [int]$_.Groups['used'].Value -gt 0 }).Count) {
        throw 'A CPU-only benchmark reported GPU selection or offload.'
    }
} else {
    $other = @($selected | Where-Object { $_.Groups['id'].Value -cne $device -or $_.Groups['name'].Value -cne $description })
    $partial = @($offloads | Where-Object { [int]$_.Groups['used'].Value -le 0 -or $_.Groups['used'].Value -ne $_.Groups['total'].Value })
    $buffer = [regex]::Match($run.Stderr, "(?im)\b$device model buffer size\s*=\s*(?<size>[0-9]+(?:\.[0-9]+)?) MiB\b")
    if (-not $selected.Count -or $other.Count -or -not $offloads.Count -or $partial.Count -or
        -not $buffer.Success -or [double]$buffer.Groups['size'].Value -le 0) {
        throw "No complete device and model offload evidence for $device ($description)."
    }
    $offloadedLayers = [int]$offloads[0].Groups['used'].Value
    $totalLayers = [int]$offloads[0].Groups['total'].Value
    $modelBufferMiB = [double]$buffer.Groups['size'].Value
}

[pscustomobject]@{
    PSTypeName = 'Pcai.LlamaDeviceBenchmark'
    Schema = 'pcai.llama-device-benchmark/v1'
    Route = $Route
    Device = $device
    DeviceDescription = $description
    CudaComputeCapability = if ($cudaCapability) { $cudaCapability.ToString() } else { $null }
    RequestedMmqPrecision = $childPrecision
    BackendSidecar = $backendSidecar
    BuildCommit = [string]$rows[0].build_commit
    BuildNumber = [int]$rows[0].build_number
    BenchSha256 = $BenchSha256.ToLowerInvariant()
    ModelSha256 = $ModelSha256.ToLowerInvariant()
    OffloadedLayers = $offloadedLayers
    TotalLayers = $totalLayers
    DeviceModelBufferMiB = $modelBufferMiB
    Parameters = [pscustomobject]@{
        PromptTokens = $PromptTokens
        DecodeTokens = $DecodeTokens
        Repetitions = $Repetitions
        Threads = $Threads
    }
    Prefill = $measurements.Prefill
    Decode = $measurements.Decode
}
