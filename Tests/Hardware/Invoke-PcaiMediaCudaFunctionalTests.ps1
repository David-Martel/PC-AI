#Requires -Version 7.0
<#
.SYNOPSIS
Exercises generation and image understanding against a local CUDA media server.

.DESCRIPTION
Starts the selected pcai-media executable on a temporary loopback port and
loads the selected local model. Checks PNG generation and nonempty image
understanding responses. Writes an isolated JSON receipt and server logs.
A missing prerequisite, startup timeout, failed request or empty suite fails
the invocation. Reported timings are functional observations for that run.

.PARAMETER ServerPath
Path to the staged or installed pcai-media executable built with CUDA.

.PARAMETER ModelPath
Path to the local Janus model directory. Run separately for each checkpoint.

.PARAMETER Device
The server's explicit CUDA selector, such as cuda:auto or cuda:0.

.PARAMETER ReadyTimeoutSeconds
Maximum startup and model load wait. Default 180 seconds.

.PARAMETER RequestTimeoutSeconds
Maximum wait for each inference request. Default 180 seconds.

.EXAMPLE
pwsh -NoProfile -File Tests/Hardware/Invoke-PcaiMediaCudaFunctionalTests.ps1 -ServerPath ./bin/pcai-media-server.exe -ModelPath ./Models/Janus-Pro-1B -Device cuda:0

.LINK
../Functional/Media.Functional.Tests.ps1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ServerPath,
    [Parameter(Mandatory)][string]$ModelPath,
    [ValidatePattern('^cuda(?::(?:auto|[0-9]+))?$')][string]$Device = 'cuda:auto',
    [ValidateRange(1, 600)][int]$ReadyTimeoutSeconds = 180,
    [ValidateRange(1, 900)][int]$RequestTimeoutSeconds = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ServerPath -PathType Leaf)) {
    throw "CUDA media server executable is missing: $ServerPath"
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Container)) {
    throw "Local Janus model directory is missing: $ModelPath"
}
$serverFile = (Get-Item -LiteralPath $ServerPath).FullName
$modelDirectory = (Get-Item -LiteralPath $ModelPath).FullName
$artifact = Join-Path ([IO.Path]::GetTempPath()) ('pcai-media-functional-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -Path $artifact -ItemType Directory
$receiptPath = Join-Path $artifact 'result.json'

function Get-LoopbackPort {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    try {
        $listener.Start()
        $listener.LocalEndpoint.Port
    } finally {
        $listener.Stop()
    }
}

function Wait-MediaModelReady {
    param([Diagnostics.Process]$Server, [string]$BaseUri, [int]$TimeoutSeconds)

    $clock = [Diagnostics.Stopwatch]::StartNew()
    while ($clock.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($Server.HasExited) { throw "Media server exited before readiness with code $($Server.ExitCode)." }
        $remainingSeconds = [math]::Ceiling($TimeoutSeconds - $clock.Elapsed.TotalSeconds)
        $probeTimeout = [int][math]::Max(1, [math]::Min(2, $remainingSeconds))
        $health = $null
        try {
            $health = Invoke-RestMethod -Uri "$BaseUri/health" -NoProxy -MaximumRedirection 0 -TimeoutSec $probeTimeout
        } catch {
            Write-Verbose "Waiting for local health endpoint. $($_.Exception.Message)"
        }
        if ($health -and $health.PSObject.Properties['model'] -and $health.model -ceq $modelDirectory) {
            if ($health.PSObject.Properties['model_loaded'] -and $health.model_loaded -eq $true -and -not $Server.HasExited) { return }
            if ($health.PSObject.Properties['status'] -and $health.status -eq 'not_ready') {
                throw 'Media server became reachable but failed to load the requested checkpoint.'
            }
        }
        Start-Sleep -Milliseconds 150
    }
    throw "Media server did not become ready within $TimeoutSeconds seconds."
}

# Protects the media server's public generation and understanding workflows.
# Detects malformed PNG responses, empty text replies, and request failures.
# Needs a local CUDA build, a usable NVIDIA GPU and an enrolled Janus checkpoint.
# Breadcrumb: Native/pcai_core/pcai_media_server/src/main.rs image handlers.
$generation = [ordered]@{
    simple = 'A red apple on a white table'
    complex_scene = 'A steampunk laboratory with brass gears, glass tubes filled with glowing green liquid, and a Victorian-era scientist examining a holographic display, dramatic lighting from gas lamps'
    technical = 'A cross-section diagram of a GPU chip showing CUDA cores arranged in streaming multiprocessors, in a technical blueprint style'
    abstract = 'Artificial intelligence represented as a neural network of glowing nodes connected by energy beams, floating in a cosmic void with nebulae in the background'
    photorealistic = 'A professional photograph of a custom-built PC with RGB lighting, visible through a tempered glass side panel, with dual monitors showing code and a shallow depth of field'
}
$understanding = [ordered]@{
    identification = 'What is the primary subject of this image? Describe it in one sentence.'
    colors = 'List the distinct colors visible in this image.'
    spatial = 'Describe the spatial layout of this image. What is in the foreground, middle ground, and background?'
    labels = 'What labels or tags would be appropriate for this image in a dataset?'
    creative = 'Write a one-paragraph creative story inspired by this image.'
    counting = 'Count and list every distinct object or element you can identify in this image.'
}
# A self-contained 2x2 RGB red PNG fixture.
$testImage = 'iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAE0lEQVR4nGP4z8DwnwGM/zMwAAAf7gP9NRsAMwAAAABJRU5ErkJggg=='
$cases = @(
    foreach ($entry in $generation.GetEnumerator()) {
        @{ Name = "generate/$($entry.Key)"; Operation = 'generate'; Payload = @{ prompt = $entry.Value; cfg_scale = 5.0; temperature = 1.0 } }
    }
    foreach ($entry in $understanding.GetEnumerator()) {
        @{
            Name = "understand/$($entry.Key)"
            Operation = 'understand'
            Payload = @{ image_base64 = $testImage; prompt = $entry.Value; max_tokens = 256; temperature = 0.1 }
        }
    }
)

$started = [DateTimeOffset]::UtcNow
$results = [Collections.Generic.List[object]]::new()
$server = $null
$stdout = $null
$stderr = $null
try {
    $port = Get-LoopbackPort
    $baseUri = "http://127.0.0.1:$port"
    $start = [Diagnostics.ProcessStartInfo]::new($serverFile)
    $start.WorkingDirectory = $artifact
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($argument in @('--host', '127.0.0.1', '--port', "$port", '--model', $modelDirectory, '--device', $Device)) {
        $start.ArgumentList.Add($argument)
    }
    $server = [Diagnostics.Process]::Start($start)
    $stdout = $server.StandardOutput.ReadToEndAsync()
    $stderr = $server.StandardError.ReadToEndAsync()
    Wait-MediaModelReady -Server $server -BaseUri $baseUri -TimeoutSeconds $ReadyTimeoutSeconds

    foreach ($case in $cases) {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $row = [ordered]@{ Case = $case.Name; Status = 'ERROR'; ElapsedMs = $null; Error = $null }
        try {
            $response = Invoke-RestMethod -Uri "$baseUri/v1/images/$($case.Operation)" -NoProxy -MaximumRedirection 0 `
                -Method Post -ContentType 'application/json' -Body ($case.Payload | ConvertTo-Json -Compress) -TimeoutSec $RequestTimeoutSeconds
            $row.Status = 'FAIL'
            if ($case.Operation -eq 'generate') {
                $row.Width = $response.width
                $row.Height = $response.height
                $image = [Convert]::FromBase64String($response.image_base64)
                if ($response.width -gt 0 -and $response.height -gt 0 -and $image.Length -ge 33 -and
                    [BitConverter]::ToString($image, 0, 8) -eq '89-50-4E-47-0D-0A-1A-0A') {
                    $row.Status = 'PASS'
                } else {
                    $row.Error = 'Generation did not return a PNG with positive dimensions.'
                }
            } else {
                $row.ResponseLength = ([string]$response.text).Length
                if (-not [string]::IsNullOrWhiteSpace($response.text)) {
                    $row.Status = 'PASS'
                } else {
                    $row.Error = 'Image understanding returned no text.'
                }
            }
        } catch {
            $row.Status = 'ERROR'
            $row.Error = $_.Exception.Message
        } finally {
            $row.ElapsedMs = $clock.Elapsed.TotalMilliseconds
            $results.Add([pscustomobject]$row)
        }
    }
} catch {
    $results.Add([pscustomobject]@{ Case = 'startup'; Status = 'ERROR'; Error = $_.Exception.Message })
} finally {
    if ($server) {
        try {
            if (-not $server.HasExited) {
                try { $server.Kill($true) } catch [InvalidOperationException] {
                    if (-not $server.HasExited) { throw }
                }
            }
            if (-not $server.WaitForExit(5000)) { throw 'Media server did not exit within five seconds.' }
            foreach ($log in @(@{ Task = $stdout; File = 'server-stdout.log' }, @{ Task = $stderr; File = 'server-stderr.log' })) {
                if ($log.Task -and $log.Task.Wait(1000)) {
                    [IO.File]::WriteAllText((Join-Path $artifact $log.File), $log.Task.Result)
                } else {
                    throw "Could not capture $($log.File) from the media server."
                }
            }
        } catch {
            $results.Add([pscustomobject]@{ Case = 'cleanup'; Status = 'ERROR'; Error = $_.Exception.Message })
        } finally {
            $server.Dispose()
        }
    }
}

$passed = @($results | Where-Object Status -eq 'PASS').Count
$success = $cases.Count -gt 0 -and $results.Count -eq $cases.Count -and $passed -eq $cases.Count
$receipt = [pscustomobject]@{
    Schema = 'pcai.media.cuda-functional/v1'
    Machine = [Environment]::MachineName
    StartedUtc = $started.ToString('o')
    CompletedUtc = [DateTimeOffset]::UtcNow.ToString('o')
    ServerPath = $serverFile
    ModelPath = $modelDirectory
    RequestedDevice = $Device
    ExpectedCases = $cases.Count
    PassedCases = $passed
    Passed = $success
    Cases = $results.ToArray()
}
[IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 5))
if (-not $success) {
    throw "CUDA media functional checks passed $passed of $($cases.Count). See $receiptPath"
}
$receiptPath
