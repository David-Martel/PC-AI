#Requires -Version 7.0
<#
.SYNOPSIS
    Captures a bounded ETW trace for Sensel haptic touchpad / HID-over-I2C repros.

.DESCRIPTION
    Starts short-lived logman trace sessions for the relevant Windows input stack:
    HIDI2C, HIDCLASS, Intel I2C, UMDF/WDF, and optional HIDI2C WPP traces.  The
    tool is designed for reproducing pointer freeze, stuck press, haptic click
    layering, or TrackPoint + palm interaction issues.

    It writes ETL files plus a JSON manifest and restores by stopping/deleting
    the trace sessions in a finally block.  It does not change registry values,
    drivers, devices, services, or touchpad settings.

.PARAMETER DurationSeconds
    Capture duration. Default: 90 seconds.

.PARAMETER OutDir
    Output root. Default: Reports\haptic-touchpad\trace-<timestamp>.

.PARAMETER Note
    Free-text scenario note written to manifest.json.

.PARAMETER IncludeHidi2cWpp
    Also capture the HIDI2C WPP provider GUID documented by Microsoft.

.PARAMETER NoCountdown
    Suppress per-second countdown output.

.EXAMPLE
    pwsh -File .\Start-HapticTouchpadTrace.ps1 -DurationSeconds 90 -IncludeHidi2cWpp -Note "TrackPoint plus palm repro"

.NOTES
    Requires elevation for logman kernel/driver provider capture.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateRange(5, 900)] [int]$DurationSeconds = 90,
    [string]$OutDir,
    [string]$Note = '',
    [switch]$IncludeHidi2cWpp,
    [switch]$NoCountdown
)

$ErrorActionPreference = 'Stop'

function Test-IsElevated {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory)] [string]$FileName,
        [Parameter(Mandatory)] [string[]]$Arguments,
        [Parameter(Mandatory)] [string]$LogPath,
        [switch]$IgnoreExitCode
    )
    $line = "$FileName " + (($Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ')
    Add-Content -Path $LogPath -Value ">>> $line"
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FileName
    foreach ($arg in $Arguments) { [void]$psi.ArgumentList.Add($arg) }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $p = [System.Diagnostics.Process]::Start($psi)
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    Add-Content -Path $LogPath -Value $stdout
    Add-Content -Path $LogPath -Value $stderr
    Add-Content -Path $LogPath -Value "<<< exit=$($p.ExitCode)`n"
    if (-not $IgnoreExitCode -and $p.ExitCode -ne 0) {
        throw "$line failed with exit code $($p.ExitCode). See $LogPath"
    }
    [pscustomobject]@{ ExitCode = $p.ExitCode; StdOut = $stdout; StdErr = $stderr }
}

function Get-TraceOutputFiles {
    param(
        [Parameter(Mandatory)] [string]$RequestedPath
    )

    $directory = Split-Path -Parent $RequestedPath
    $leaf = Split-Path -Leaf $RequestedPath
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($leaf)
    $extension = [System.IO.Path]::GetExtension($leaf)

    $matches = @()
    if (Test-Path -LiteralPath $RequestedPath) {
        $matches += Get-Item -LiteralPath $RequestedPath
    }
    if (Test-Path -LiteralPath $directory) {
        $matches += Get-ChildItem -LiteralPath $directory -File -Filter "$stem*$extension" -ErrorAction SilentlyContinue
    }

    @($matches |
        Sort-Object -Property FullName -Unique |
        Select-Object FullName, Length, LastWriteTime)
}

if (-not (Test-IsElevated)) {
    throw 'Start-HapticTouchpadTrace.ps1 must run as Administrator.'
}

if (-not $OutDir) {
    $repo = Resolve-Path (Join-Path $PSScriptRoot '..\..') | Select-Object -ExpandProperty Path
    $OutDir = Join-Path $repo ("Reports\haptic-touchpad\trace-{0}" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$commandLog = Join-Path $OutDir 'logman-commands.txt'
$manifestPath = Join-Path $OutDir 'manifest.json'
$session = 'PC_AI_HapticTouchpad_ETW'
$wppSession = 'PC_AI_HIDI2C_WPP'
$etl = Join-Path $OutDir 'haptic-touchpad.etl'
$wppEtl = Join-Path $OutDir 'hidi2c-wpp.etl'

$providers = @(
    @{ Name = 'Microsoft-Windows-SPB-HIDI2C'; Flags = '0xFFFFFFFF'; Level = '5' },
    @{ Name = 'Microsoft-Windows-Input-HIDCLASS'; Flags = '0xFFFFFFFF'; Level = '5' },
    @{ Name = 'Intel-iaLPSS2-I2C'; Flags = '0xFFFFFFFF'; Level = '5' },
    @{ Name = 'Intel-iaLPSS-I2C'; Flags = '0xFFFFFFFF'; Level = '5' },
    @{ Name = '{485E7DE9-0A80-11D8-AD15-505054503030}'; Flags = '0xFFFFFFFF'; Level = '5' }
)

$started = [System.Collections.Generic.List[string]]::new()
$startTime = Get-Date

try {
    if ($PSCmdlet.ShouldProcess($session, "Create ETW trace at $etl")) {
        Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('delete', $session, '-ets') -LogPath $commandLog -IgnoreExitCode | Out-Null
        Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('create', 'trace', $session, '-o', $etl, '-nb', '128', '640', '-bs', '128', '-ow') -LogPath $commandLog | Out-Null
        foreach ($provider in $providers) {
            Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('update', 'trace', $session, '-p', $provider.Name, $provider.Flags, $provider.Level) -LogPath $commandLog | Out-Null
        }
        Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('start', $session) -LogPath $commandLog | Out-Null
        [void]$started.Add($session)
    }

    if ($IncludeHidi2cWpp -and $PSCmdlet.ShouldProcess($wppSession, "Create HIDI2C WPP trace at $wppEtl")) {
        Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('delete', $wppSession, '-ets') -LogPath $commandLog -IgnoreExitCode | Out-Null
        Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('create', 'trace', $wppSession, '-o', $wppEtl, '-nb', '128', '640', '-bs', '128', '-ow') -LogPath $commandLog | Out-Null
        Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('update', 'trace', $wppSession, '-p', '{E742C27D-29B1-4E4B-94EE-074D3AD72836}', '0x7FFFFFFF', '255') -LogPath $commandLog | Out-Null
        Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('start', $wppSession) -LogPath $commandLog | Out-Null
        [void]$started.Add($wppSession)
    }

    Write-Host "Trace running for $DurationSeconds seconds." -ForegroundColor Cyan
    Write-Host 'Reproduce the haptic touchpad issue now: TrackPoint movement + palm/press/click pattern.' -ForegroundColor Yellow
    for ($remaining = $DurationSeconds; $remaining -gt 0; $remaining--) {
        if (-not $NoCountdown -and ($remaining -eq $DurationSeconds -or $remaining % 10 -eq 0 -or $remaining -le 5)) {
            Write-Host "  remaining: $remaining s"
        }
        Start-Sleep -Seconds 1
    }
} finally {
    $stopTime = Get-Date
    foreach ($name in @($started | Sort-Object -Descending)) {
        try { Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('stop', $name, '-ets') -LogPath $commandLog -IgnoreExitCode | Out-Null } catch {}
        try { Invoke-LoggedCommand -FileName 'logman.exe' -Arguments @('delete', $name) -LogPath $commandLog -IgnoreExitCode | Out-Null } catch {}
    }

    $deviceSnapshot = @()
    try {
        $deviceSnapshot = @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
            Where-Object { $_.InstanceId -match 'SNSL002D|LEN032A|VEN_8086&DEV_7E78|ELAS|ETDHSA' -or $_.FriendlyName -match 'Touch|TrackPoint|ELAN|I2C|Sensel' } |
            Select-Object Class, FriendlyName, InstanceId, Status, Problem)
    } catch {}

    $precisionSettings = @()
    foreach ($key in 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\PrecisionTouchPad', 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\PrecisionTouchPad') {
        if (Test-Path -LiteralPath $key) {
            $precisionSettings += [pscustomobject]@{
                Path = $key
                Values = Get-ItemProperty -Path $key | Select-Object * -ExcludeProperty PSPath, PSParentPath, PSChildName, PSDrive, PSProvider
            }
        }
    }

    $etlFiles = @(Get-TraceOutputFiles -RequestedPath $etl)
    $wppEtlFiles = if ($IncludeHidi2cWpp) { @(Get-TraceOutputFiles -RequestedPath $wppEtl) } else { @() }

    $manifest = [ordered]@{
        Timestamp = (Get-Date).ToString('o')
        Machine = $env:COMPUTERNAME
        Note = $Note
        StartTime = $startTime.ToString('o')
        StopTime = $stopTime.ToString('o')
        DurationSeconds = [math]::Round(($stopTime - $startTime).TotalSeconds, 1)
        OutDir = $OutDir
        Etl = $etl
        EtlFiles = $etlFiles
        WppEtl = if ($IncludeHidi2cWpp) { $wppEtl } else { $null }
        WppEtlFiles = $wppEtlFiles
        Providers = $providers
        DeviceSnapshot = $deviceSnapshot
        PrecisionTouchpadSettings = $precisionSettings
        CommandLog = $commandLog
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding UTF8
    Write-Host "Trace manifest written: $manifestPath" -ForegroundColor Green
}
