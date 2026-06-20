#Requires -Version 7.0
<#
.SYNOPSIS
    Reports NVIDIA internal/eGPU driver health without changing system state.

.DESCRIPTION
    Captures the local NVIDIA display adapters, DriverStore display packages,
    nvidia-smi state, and obvious driver-version mismatch signals. This is a
    read-only diagnostic intended for fragile ThinkPad plus eGPU configurations.

.PARAMETER OutputDirectory
    Directory where JSON and text evidence should be written.

.PARAMETER AsJson
    Emit the summary object as JSON to stdout.

.PARAMETER FailOnIssue
    Exit with code 1 when a driver split, NVIDIA device problem, or missing
    NVIDIA update surface is detected. Use in automation to fail loudly.
#>
param(
    [string]$OutputDirectory = (Join-Path (Resolve-Path "$PSScriptRoot\..\..").Path 'Logs\nvidia-dual-gpu'),
    [switch]$AsJson,
    [switch]$FailOnIssue
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$runDir = Join-Path $OutputDirectory $stamp
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Invoke-CaptureText {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [scriptblock] $ScriptBlock
    )

    $path = Join-Path $runDir $Name
    try {
        & $ScriptBlock *> $path
    } catch {
        $_ | Out-String | Set-Content -LiteralPath $path -Encoding UTF8
    }
    $path
}

function Get-NvidiaSmiDriverVersion {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        return $null
    }

    $raw = & $smi.Source 2>$null | Out-String
    if ($raw -match 'Driver Version:\s+([0-9.]+)') {
        return $Matches[1]
    }
    $null
}

function Get-NvidiaLocalApplications {
    $appRoot = 'C:\Program Files\NVIDIA Corporation\NVIDIA App'
    $updateRoot = 'C:\ProgramData\NVIDIA Corporation\NVIDIA app\UpdateFramework'
    $processes = @(Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessName -match '^(NVIDIA App|NVIDIA Overlay|nvcontainer|NVDisplay\.Container|nvWmi64)$' } |
        Select-Object ProcessName, Id, Path)

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    $artifacts = @()
    if (Test-Path -LiteralPath $updateRoot) {
        $artifacts = @(Get-ChildItem -LiteralPath $updateRoot -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '\.(exe|inf|json)$' } |
            Select-Object -First 25 FullName, Length, LastWriteTime)
    }

    [pscustomobject]@{
        NvidiaAppRoot = $appRoot
        NvidiaAppInstalled = Test-Path -LiteralPath $appRoot
        UpdateFrameworkRoot = $updateRoot
        UpdateFrameworkPresent = Test-Path -LiteralPath $updateRoot
        UpdateArtifactSample = $artifacts
        WingetAvailable = [bool]$winget
        Processes = $processes
    }
}

$pnpDisplay = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
    Select-Object Status, Class, FriendlyName, InstanceId, Problem, ConfigManagerErrorCode

$videoControllers = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
    Select-Object Name, PNPDeviceID, DriverVersion, Status, AdapterRAM

$nvidiaVideo = @($videoControllers | Where-Object { $_.Name -like '*NVIDIA*' })
$nvidiaVersions = @($nvidiaVideo | ForEach-Object { $_.DriverVersion } | Where-Object { $_ } | Sort-Object -Unique)
$nvidiaProblems = @($pnpDisplay | Where-Object { $_.FriendlyName -like '*NVIDIA*' -and $_.Status -ne 'OK' })
$smiVersion = Get-NvidiaSmiDriverVersion
$localApplications = Get-NvidiaLocalApplications

$issues = [System.Collections.Generic.List[string]]::new()
if (@($nvidiaVersions).Count -gt 1) {
    $issues.Add("NVIDIA display adapters report more than one driver version: $($nvidiaVersions -join ', ')")
}
if (@($nvidiaProblems).Count -gt 0) {
    foreach ($problem in $nvidiaProblems) {
        $issues.Add("NVIDIA device problem: $($problem.FriendlyName) status=$($problem.Status) code=$($problem.ConfigManagerErrorCode)")
    }
}
if ($smiVersion -and @($nvidiaVersions).Count -gt 0 -and ($nvidiaVersions -notcontains $smiVersion)) {
    $issues.Add("nvidia-smi reports driver $smiVersion, which does not match all Win32_VideoController versions.")
}
if (-not $localApplications.NvidiaAppInstalled -and -not $localApplications.WingetAvailable) {
    $issues.Add('No local NVIDIA App install and no winget command were detected for update orchestration.')
}

$recommendedActions = @(
    'Do not run a blind NVIDIA installer while the internal GPU and eGPU report different driver/package state.',
    'Before any install, extract the candidate NVIDIA package and verify its Display.Driver INF covers both VEN_10DE&DEV_28B8 and VEN_10DE&DEV_2D04.',
    'After any driver change, rerun this script with -FailOnIssue and compare nvidia-smi with Win32_VideoController and pnputil problem devices.'
)

$pnputilProblemsPath = Invoke-CaptureText 'pnputil-problems.txt' { pnputil.exe /enum-devices /problem }
$pnputilDisplayDriversPath = Invoke-CaptureText 'pnputil-display-drivers.txt' { pnputil.exe /enum-drivers /class Display }
$nvidiaSmiPath = Invoke-CaptureText 'nvidia-smi.txt' { nvidia-smi }
$nvidiaSmiQueryPath = Invoke-CaptureText 'nvidia-smi-q.txt' { nvidia-smi -q }

$summary = [pscustomobject]@{
    Timestamp = Get-Date
    Machine = $env:COMPUTERNAME
    RunDirectory = $runDir
    NvidiaDriverVersions = $nvidiaVersions
    NvidiaSmiDriverVersion = $smiVersion
    HasNvidiaDriverVersionSplit = (@($nvidiaVersions).Count -gt 1)
    HasNvidiaDeviceProblem = (@($nvidiaProblems).Count -gt 0)
    HasIssues = ($issues.Count -gt 0)
    Issues = @($issues)
    RecommendedActions = $recommendedActions
    LocalApplications = $localApplications
    NvidiaProblems = $nvidiaProblems
    NvidiaVideoControllers = $nvidiaVideo
    Evidence = [pscustomobject]@{
        PnputilProblems = $pnputilProblemsPath
        PnputilDisplayDrivers = $pnputilDisplayDriversPath
        NvidiaSmi = $nvidiaSmiPath
        NvidiaSmiQuery = $nvidiaSmiQueryPath
    }
}

$summary | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $runDir 'summary.json') -Encoding UTF8

if ($AsJson) {
    $summary | ConvertTo-Json -Depth 8
} else {
    $summary | Format-List
}

if ($FailOnIssue -and $summary.HasIssues) {
    exit 1
}
