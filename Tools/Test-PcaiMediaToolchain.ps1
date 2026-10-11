#Requires -Version 7.0

<#
.SYNOPSIS
Qualifies installed Windows media tools without opening capture devices.
.DESCRIPTION
Default operation enumerates compiled FFmpeg backends and fresh Windows GPU
inventory. Runtime qualification is separate: ValidateIntelQsv encodes exactly
16 synthetic 320x240 frames to the null muxer. It reads no camera, microphone,
file or network stream. No NVIDIA encode is attempted. Python is inspected only
when an explicit virtual-environment interpreter is supplied. All subprocesses
have a timeout and retain exit codes and errors; missing tools remain unknown.
.PARAMETER PythonPath
Absolute path to an existing venv Scripts/python.exe with a pyvenv.cfg parent.
.PARAMETER GStreamerPath
Optional absolute gst-inspect-1.0.exe path; current and older defaults are searched.
.PARAMETER ValidateIntelQsv
Run a small synthetic Intel QSV encode to verify runtime availability.
.PARAMETER ListCaptureDevices
List DirectShow device names using FFmpeg's dummy input; never open a real device.
.PARAMETER OutputPath
Optional absolute JSON report path. Existing reports are never overwritten.
.PARAMETER TimeoutSeconds
Timeout for each external command, default 15 seconds.
.PARAMETER DryRun
Show intended operations without provider queries, subprocesses or writes.
.PARAMETER Help
Show help; accepts -h and --help.
.EXAMPLE
pwsh -NoLogo -NoProfile -File .\Tools\Test-PcaiMediaToolchain.ps1 -DryRun
.EXAMPLE
pwsh -NoLogo -NoProfile -NonInteractive -File .\Tools\Test-PcaiMediaToolchain.ps1 -ValidateIntelQsv
#>
[CmdletBinding()]
param(
    [string]$PythonPath,
    [string]$GStreamerPath,
    [switch]$ValidateIntelQsv,
    [switch]$ListCaptureDevices,
    [string]$OutputPath,
    [ValidateRange(1, 120)] [int]$TimeoutSeconds = 15,
    [switch]$DryRun,
    [Alias('h', '-help')] [switch]$Help
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-MediaReportPath {
    <#.SYNOPSIS
    Validates an explicit report path before any writes.
    #>
    param([string]$Path)
    if (-not [System.IO.Path]::IsPathFullyQualified($Path)) { throw 'OutputPath must be absolute.' }
    $resolved = [System.IO.Path]::GetFullPath($Path)
    foreach ($part in $resolved.Split([char[]]'\/')) {
        $leaf = $part.TrimEnd('.', ' ')
        if ($leaf -eq '$null' -or $leaf -match '^(?i:AUX|CON|NUL|PRN|COM[1-9]|LPT[1-9])(?:\.|$)') {
            throw 'OutputPath contains a reserved Windows component.'
        }
    }
    if ($resolved.Substring(2).Contains(':') -or [string]::IsNullOrEmpty([System.IO.Path]::GetFileName($resolved))) {
        throw 'OutputPath must name a file without alternate stream syntax.'
    }
    if ([System.IO.File]::Exists($resolved) -or [System.IO.Directory]::Exists($resolved)) {
        throw 'OutputPath already exists; choose a new report name.'
    }
    if (-not [System.IO.Directory]::Exists([System.IO.Path]::GetDirectoryName($resolved))) {
        throw 'OutputPath parent directory must already exist.'
    }
    return $resolved
}

function Invoke-MediaProcess {
    <#.SYNOPSIS
    Runs one external command with asynchronous pipe drains and a hard timeout.
    #>
    param([string]$FilePath, [string[]]$Arguments, [int]$TimeoutSeconds = 15)
    $result = [ordered]@{
        file = $FilePath; arguments = $Arguments; status = 'Failed'
        exitCode = $null; timedOut = $false; stdout = ''; stderr = ''
        error = $null; elapsedMilliseconds = 0
    }
    $process = [System.Diagnostics.Process]::new()
    $started = $false
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $info = [System.Diagnostics.ProcessStartInfo]::new()
        $info.FileName = $FilePath
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
        $process.StartInfo = $info
        if (-not $process.Start()) { throw 'Process did not start.' }
        $started = $true
        $outTask = $process.StandardOutput.ReadToEndAsync()
        $errTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $result.timedOut = $true
            $result.status = 'TimedOut'
            $process.Kill($true)
            if (-not $process.WaitForExit(3000)) { throw 'Timed-out child did not terminate.' }
        }
        # A descendant can inherit a pipe even after the main child exits.
        if (-not [System.Threading.Tasks.Task]::WaitAll([System.Threading.Tasks.Task[]]@($outTask, $errTask), 3000)) {
            throw 'Child output pipes did not close within three seconds.'
        }
        $result.stdout = $outTask.Result
        $result.stderr = $errTask.Result
        $result.exitCode = $process.ExitCode
        if (-not $result.timedOut) { $result.status = if ($process.ExitCode -eq 0) { 'Succeeded' } else { 'Failed' } }
    }
    catch { $result.error = $_.Exception.Message }
    finally {
        if ($started -and -not $process.HasExited) {
            try {
                $process.Kill($true)
                if (-not $process.WaitForExit(3000)) { throw 'Child cleanup did not finish.' }
            }
            catch { $result.error = "$($result.error) Cleanup: $($_.Exception.Message)" }
        }
        $result.elapsedMilliseconds = [math]::Round($timer.Elapsed.TotalMilliseconds, 1)
        $process.Dispose()
    }
    return [pscustomobject]$result
}

function Find-MediaExecutable {
    <#.SYNOPSIS
    Resolves application files without PowerShell module auto-loading.
    #>
    param([string]$Name, [string]$ExplicitPath, [string[]]$Candidates = @())
    if ($ExplicitPath) {
        if (-not [System.IO.Path]::IsPathFullyQualified($ExplicitPath) -or -not [System.IO.File]::Exists($ExplicitPath)) {
            throw "Explicit executable does not exist at an absolute path: $ExplicitPath"
        }
        return [System.IO.Path]::GetFullPath($ExplicitPath)
    }
    $command = Microsoft.PowerShell.Core\Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { return $command[0].Source }
    foreach ($candidate in $Candidates) { if ([System.IO.File]::Exists($candidate)) { return $candidate } }
    return $null
}

function Get-MediaGpuInventory {
    <#.SYNOPSIS
    Collects current Windows display controllers, retaining provider failure.
    #>
    try {
        $devices = @(CimCmdlets\Get-CimInstance -ClassName Win32_VideoController -OperationTimeoutSec 3 -ErrorAction Stop |
                Microsoft.PowerShell.Utility\Select-Object Name, DriverVersion, PNPDeviceID)
        return [pscustomobject]@{ status = 'Available'; devices = $devices; error = $null }
    }
    catch { return [pscustomobject]@{ status = 'Unknown'; devices = @(); error = $_.Exception.Message; exceptionType = $_.Exception.GetType().FullName } }
}

function Get-MediaToolchainReport {
    <#.SYNOPSIS
    Separates installed/compiled capabilities from optional bounded runtime tests.
    #>
    param([string]$PythonPath, [string]$GStreamerPath, [switch]$ValidateIntelQsv,
        [switch]$ListCaptureDevices, [int]$TimeoutSeconds = 15, [switch]$DryRun)
    if ($DryRun) {
        return [pscustomobject]@{ status = 'DryRun'; writesPerformed = $false; processesStarted = $false
            inventoryQueried = $false; intelSyntheticFrames = $(if ($ValidateIntelQsv) { 16 } else { 0 })
            listCaptureDevices = $ListCaptureDevices.IsPresent; pythonRequested = [bool]$PythonPath }
    }
    if (-not $IsWindows) { throw 'Media qualification requires Windows.' }
    if ($PythonPath) {
        if (-not [System.IO.Path]::IsPathFullyQualified($PythonPath) -or -not [System.IO.File]::Exists($PythonPath)) {
            throw 'PythonPath must point to an existing absolute venv interpreter.'
        }
        $scriptsDir = [System.IO.Path]::GetDirectoryName($PythonPath)
        $venvRoot = [System.IO.Path]::GetDirectoryName($scriptsDir)
        if ([System.IO.Path]::GetFileName($PythonPath) -ne 'python.exe' -or
            [System.IO.Path]::GetFileName($scriptsDir) -ne 'Scripts' -or
            -not [System.IO.File]::Exists([System.IO.Path]::Combine($venvRoot, 'pyvenv.cfg'))) {
            throw 'PythonPath must be an isolated venv Scripts/python.exe with pyvenv.cfg.'
        }
        $venvConfig = [System.IO.File]::ReadAllText([System.IO.Path]::Combine($venvRoot, 'pyvenv.cfg'))
        $siteSettings = [regex]::Matches($venvConfig, '(?im)^\s*include-system-site-packages\s*=\s*(\S+)\s*$')
        if ($siteSettings.Count -ne 1 -or $siteSettings[0].Groups[1].Value -ine 'false') {
            throw 'PythonPath must disable system site packages explicitly in pyvenv.cfg.'
        }
    }
    $ffmpeg = Find-MediaExecutable -Name 'ffmpeg.exe'
    $ffprobe = Find-MediaExecutable -Name 'ffprobe.exe'
    $gstCandidates = @(
        "$env:LOCALAPPDATA\Programs\gstreamer\1.0\msvc_x86_64\bin\gst-inspect-1.0.exe",
        "$env:ProgramFiles\gstreamer\1.0\msvc_x86_64\bin\gst-inspect-1.0.exe",
        'C:\gstreamer\1.0\msvc_x86_64\bin\gst-inspect-1.0.exe'
    )
    $gst = Find-MediaExecutable -Name 'gst-inspect-1.0.exe' -ExplicitPath $GStreamerPath -Candidates $gstCandidates
    $report = [ordered]@{
        schemaVersion = 1; utc = [datetimeoffset]::UtcNow.ToString('o'); status = 'Completed'
        interpretation = 'Compiled listings do not establish usable hardware; runtime tests are separate'
        gpuInventory = Get-MediaGpuInventory
        ffmpeg = [ordered]@{ path = $ffmpeg; status = 'Missing'; probes = @(); compiledHardwareMethods = @(); compiledEncoders = @() }
        ffprobe = [ordered]@{ path = $ffprobe; status = 'Missing'; probe = $null }
        gstreamer = [ordered]@{ path = $gst; status = 'Missing'; probes = @(); interpretation = 'Element metadata only; no pipeline opened' }
        intelQsv = [ordered]@{ status = 'NotRequested'; syntheticFrames = 16; width = 320; height = 240; probe = $null }
        nvidiaRuntime = [ordered]@{ status = 'NotAttempted'; reason = 'This tool does not run NVIDIA workloads' }
        captureDeviceListing = [ordered]@{ status = 'NotRequested'; probe = $null }
        python = [ordered]@{ path = $PythonPath; status = 'NotRequested'; probe = $null; data = $null }
    }
    if ($ffmpeg) {
        $probes = @(
            Invoke-MediaProcess $ffmpeg @('-hide_banner', '-version') $TimeoutSeconds
            Invoke-MediaProcess $ffmpeg @('-hide_banner', '-hwaccels') $TimeoutSeconds
            Invoke-MediaProcess $ffmpeg @('-hide_banner', '-encoders') $TimeoutSeconds
        )
        $report.ffmpeg.probes = $probes
        $report.ffmpeg.status = if (@($probes | Microsoft.PowerShell.Core\Where-Object status -ne 'Succeeded').Count -eq 0) { 'Enumerated' } else { 'Failed' }
        if ($probes[1].status -eq 'Succeeded') {
            $report.ffmpeg.compiledHardwareMethods = @($probes[1].stdout -split '\r?\n' |
                Microsoft.PowerShell.Core\Where-Object { $_ -match '^\s*(cuda|qsv|d3d11va|d3d12va|dxva2|opencl|vulkan|vaapi|amf)\s*$' } |
                Microsoft.PowerShell.Core\ForEach-Object { $_.Trim() })
        }
        if ($probes[2].status -eq 'Succeeded') {
            $report.ffmpeg.compiledEncoders = @($probes[2].stdout -split '\r?\n' |
                Microsoft.PowerShell.Core\Where-Object { $_ -match '^\s*[VA][A-Z.]{5}\s+(\S+)' } |
                Microsoft.PowerShell.Core\ForEach-Object { [regex]::Match($_, '^\s*[VA][A-Z.]{5}\s+(\S+)').Groups[1].Value })
        }
    }
    if ($ffprobe) {
        $report.ffprobe.probe = Invoke-MediaProcess $ffprobe @('-hide_banner', '-version') $TimeoutSeconds
        $report.ffprobe.status = $report.ffprobe.probe.status
    }
    if ($gst) {
        $gstProbes = [System.Collections.Generic.List[object]]::new()
        foreach ($element in @('--version', 'qsvh264enc', 'd3d11h264dec', 'nvh264enc', 'wasapi2src')) {
            $gstProbes.Add((Invoke-MediaProcess $gst @($element) $TimeoutSeconds))
        }
        $report.gstreamer.probes = $gstProbes.ToArray()
        $report.gstreamer.status = if ($gstProbes[0].status -eq 'Succeeded') { 'Enumerated' } else { 'Failed' }
    }
    if ($ValidateIntelQsv) {
        if (-not $ffmpeg) { $report.intelQsv.status = 'MissingFfmpeg' }
        elseif ($report.gpuInventory.status -ne 'Available') { $report.intelQsv.status = 'UnknownGpuInventory' }
        elseif (@($report.gpuInventory.devices | Microsoft.PowerShell.Core\Where-Object Name -match 'Intel').Count -eq 0) {
            $report.intelQsv.status = 'NoIntelGpu'
        }
        else {
            $qsvArgs = @('-nostdin', '-hide_banner', '-loglevel', 'error', '-init_hw_device', 'qsv:pcai,child_device_type=d3d11va',
                '-f', 'lavfi', '-i', 'color=c=black:s=320x240:r=30', '-frames:v', '16', '-an', '-vf', 'format=nv12',
                '-c:v', 'h264_qsv', '-async_depth', '1', '-threads', '1', '-f', 'null', '-')
            $report.intelQsv.probe = Invoke-MediaProcess $ffmpeg $qsvArgs $TimeoutSeconds
            $report.intelQsv.status = if ($report.intelQsv.probe.status -eq 'Succeeded') { 'Validated' } else { $report.intelQsv.probe.status }
        }
    }
    if ($ListCaptureDevices) {
        if (-not $ffmpeg) { $report.captureDeviceListing.status = 'MissingFfmpeg' }
        else {
            $probe = Invoke-MediaProcess $ffmpeg @('-nostdin', '-hide_banner', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy') $TimeoutSeconds
            $report.captureDeviceListing.probe = $probe
            # DirectShow deliberately exits nonzero after its metadata listing.
            $report.captureDeviceListing.status = if ($probe.timedOut -or $probe.error) { 'Failed' }
                elseif ($probe.stderr -match '(?im)\((video|audio)\)') { 'Listed' }
                elseif ($probe.stderr -match '(?im)(DirectShow|Could not enumerate)') { 'NoDevicesOrProviderFailure' }
                else { 'Failed' }
        }
    }
    if ($PythonPath) {
        $pythonCode = @'
import importlib.metadata as md, json
variants = ['opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless']
packages = {d.metadata['Name'].lower(): d.version for d in md.distributions()}
installed = {n: packages[n] for n in variants if n in packages}
result = {'opencvVariants': installed, 'opencvConflict': len(installed) > 1, 'cv2': None, 'av': None, 'errors': []}
if len(installed) == 1:
    try:
        import cv2
        result['cv2'] = {'version': cv2.__version__, 'buildInformation': cv2.getBuildInformation(), 'cudaDeviceCount': cv2.cuda.getCudaEnabledDeviceCount()}
    except Exception as e: result['errors'].append({'module': 'cv2', 'type': type(e).__name__, 'message': str(e)})
if 'av' in packages:
    try:
        import av
        result['av'] = {'version': av.__version__, 'libraryVersions': av.library_versions}
    except Exception as e: result['errors'].append({'module': 'av', 'type': type(e).__name__, 'message': str(e)})
print(json.dumps(result))
'@
        $probe = Invoke-MediaProcess $PythonPath @('-I', '-c', $pythonCode) $TimeoutSeconds
        $report.python.probe = $probe
        $report.python.status = $probe.status
        if ($probe.status -eq 'Succeeded') {
            try {
                $data = Microsoft.PowerShell.Utility\ConvertFrom-Json -InputObject $probe.stdout -ErrorAction Stop
                $report.python.data = $data
                $report.python.status = if ($data.opencvConflict) { 'OpenCvConflict' }
                    elseif (@($data.errors).Count -gt 0) { 'ImportFailed' }
                    elseif ($null -eq $data.cv2 -or $null -eq $data.av) { 'Incomplete' } else { 'Inspected' }
            }
            catch { $report.python.status = 'InvalidJson'; $report.python.data = [ordered]@{ error = $_.Exception.Message } }
        }
    }
    if ($report.ffmpeg.status -in @('Missing', 'Failed') -or $report.ffprobe.status -in @('Missing', 'Failed', 'TimedOut') -or
        $report.gstreamer.status -in @('Missing', 'Failed') -or $report.gpuInventory.status -ne 'Available' -or
        $report.python.status -in @('Failed', 'TimedOut', 'OpenCvConflict', 'ImportFailed', 'Incomplete', 'InvalidJson') -or
        ($ValidateIntelQsv -and $report.intelQsv.status -ne 'Validated') -or
        ($ListCaptureDevices -and $report.captureDeviceListing.status -ne 'Listed')) {
        $report.status = 'CompletedWithIssues'
    }
    return [pscustomobject]$report
}

if ($Help) { Microsoft.PowerShell.Core\Get-Help -Name $PSCommandPath -Detailed; return }
$validatedReportPath = if ($OutputPath) { Resolve-MediaReportPath $OutputPath } else { $null }
$report = Get-MediaToolchainReport -PythonPath $PythonPath -GStreamerPath $GStreamerPath -ValidateIntelQsv:$ValidateIntelQsv `
    -ListCaptureDevices:$ListCaptureDevices -TimeoutSeconds $TimeoutSeconds -DryRun:$DryRun
if ($validatedReportPath -and -not $DryRun) {
    $json = Microsoft.PowerShell.Utility\ConvertTo-Json -InputObject $report -Depth 12
    $stream = [System.IO.FileStream]::new($validatedReportPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
    $writer = [System.IO.StreamWriter]::new($stream, [System.Text.UTF8Encoding]::new($false))
    try { $writer.Write($json) } finally { $writer.Dispose(); $stream.Dispose() }
}
$report
