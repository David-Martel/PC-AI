<#
.SYNOPSIS
Collects UI responsiveness, OneDrive, Process Lasso, and filter-driver evidence.

.DESCRIPTION
Writes a timestamped diagnostic bundle under Reports\ui-glitch-diagnostics by
capturing process state, Process Lasso logs, OneDrive sync diagnostics, filter
state, and recent event-log evidence.

.PARAMETER SinceMinutes
Lookback window for event-log queries.

.PARAMETER OutputRoot
Root directory for timestamped diagnostic output.

.PARAMETER DryRun
Preview the output directory and capture plan without writing files.
The long CLI form `--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.
#>
[CmdletBinding()]
param(
    [int]$SinceMinutes = 30,
    [string]$OutputRoot = (Join-Path (Get-Location) 'Reports\ui-glitch-diagnostics'),
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$CliArgs = @($CliArgs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
if (@($CliArgs) -contains '--help') {
    $Help = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--help' })
}
if (@($CliArgs) -contains '--DryRun') {
    $DryRun = $true
    $CliArgs = @($CliArgs | Where-Object { $_ -ne '--DryRun' })
}
if ($Help) {
    $helpMatch = [regex]::Match((Get-Content -LiteralPath $PSCommandPath -Raw), '(?s)<#\s*(.*?)\s*#>')
    if ($helpMatch.Success) { $helpMatch.Groups[1].Value.Trim() } else { Get-Help -Detailed $PSCommandPath }
    return
}
if (@($CliArgs).Count -gt 0) {
    throw "Unknown CLI argument(s): $($CliArgs -join ', ')"
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outDir = Join-Path $OutputRoot $stamp
if ($DryRun) {
    [pscustomobject]@{
        DryRun = $true
        PlannedOutputDirectory = $outDir
        SinceMinutes = $SinceMinutes
        Captures = @(
            'process-snapshot'
            'onedrive-process-io'
            'filter-state'
            'processlasso-logs'
            'onedrive-syncdiagnostics-tail'
            'system-storage-filter-events'
            'application-wer-hang-events'
        )
    }
    return
}
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

function Write-Section {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [scriptblock]$ScriptBlock
    )

    try {
        & $ScriptBlock | Out-File -LiteralPath $Path -Encoding UTF8
    } catch {
        "ERROR: $($_.Exception.Message)" | Out-File -LiteralPath $Path -Encoding UTF8
    }
}

$processNames = @(
    'ProcessLasso',
    'ProcessGovernor',
    'bitsumsessionagent',
    'OneDrive',
    'OneDrive.Sync.Service',
    'FileSyncHelper',
    'explorer',
    'dwm',
    'TextInputHost',
    'TabTip',
    'SynRpcServer',
    'Lenovo.Modern.ImController',
    'LenovoVantageService'
)

Write-Section (Join-Path $outDir 'process-snapshot.txt') {
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $processNames -contains $_.ProcessName } |
        Select-Object Id, ProcessName, Responding, StartTime, CPU, WorkingSet64, Handles, Path |
        Sort-Object ProcessName, Id |
        Format-List
}

Write-Section (Join-Path $outDir 'onedrive-process-io.txt') {
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -in @('OneDrive.exe', 'OneDrive.Sync.Service.exe', 'FileSyncHelper.exe') } |
        Select-Object Name, ProcessId, WorkingSetSize, ReadOperationCount, WriteOperationCount, ReadTransferCount, WriteTransferCount, CommandLine |
        Format-List
}

Write-Section (Join-Path $outDir 'filter-state.txt') {
    fltmc filters
    ''
    fltmc volumes
}

$plLogDir = 'C:\ProgramData\ProcessLasso\logs'
if (Test-Path $plLogDir) {
    $plOut = Join-Path $outDir 'processlasso-logs'
    New-Item -ItemType Directory -Path $plOut -Force | Out-Null
    Get-ChildItem -LiteralPath $plLogDir -Filter 'processlasso.log*' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 4 |
        Copy-Item -Destination $plOut -Force
}

$oneDriveDiag = Join-Path $env:LOCALAPPDATA 'Microsoft\OneDrive\logs\Personal\SyncDiagnostics.log'
if (Test-Path $oneDriveDiag) {
    Get-Content -LiteralPath $oneDriveDiag -Tail 250 |
        Set-Content -LiteralPath (Join-Path $outDir 'onedrive-syncdiagnostics-tail.txt') -Encoding UTF8
}

$ms = [Math]::Max(1, $SinceMinutes) * 60 * 1000
$eventQueries = @{
    'system-storage-filter-events.txt' = "*[System[(Provider[@Name='Microsoft-Windows-FilterManager'] or Provider[@Name='Microsoft-Windows-Ntfs'] or Provider[@Name='disk'] or Provider[@Name='CldFlt']) and TimeCreated[timediff(@SystemTime) <= $ms]]]"
    'application-wer-hang-events.txt' = "*[System[((Provider[@Name='Windows Error Reporting'] or Provider[@Name='Application Hang']) or EventID=1002) and TimeCreated[timediff(@SystemTime) <= $ms]]]"
}

foreach ($item in $eventQueries.GetEnumerator()) {
    $logName = if ($item.Key -like 'system-*') { 'System' } else { 'Application' }
    $target = Join-Path $outDir $item.Key
    & wevtutil.exe qe $logName "/q:$($item.Value)" /rd:true /c:200 /f:text |
        Out-File -LiteralPath $target -Encoding UTF8
}

[pscustomobject]@{
    OutputDirectory = $outDir
    SinceMinutes = $SinceMinutes
    CapturedAt = (Get-Date).ToString('o')
}
