#Requires -Version 7.0
<#
.SYNOPSIS
Validates Process Lasso boot-safety exclusions and logging settings.

.DESCRIPTION
Checks that ProcessGovernor.exe is running, expected ProBalance and SmartTrim
exclusions are configured, logging flags are enabled, and recent Process Lasso
log lines are available for boot or UI-glitch investigations.

.PARAMETER ConfigPath
Path to the Process Lasso INI file to validate.

.PARAMETER LogPath
Path to the Process Lasso log file.

.PARAMETER ReportPath
Optional JSON report path. It is not written when DryRun is set.

.PARAMETER DryRun
Run the validation without writing the optional report file. The long CLI form
`--DryRun` is also accepted.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.
#>
[CmdletBinding()]
param(
    [string]$ConfigPath = 'C:\ProgramData\ProcessLasso\config\prolasso.ini',

    [string]$LogPath = 'C:\ProgramData\ProcessLasso\logs\processlasso.log',

    [int]$LookbackMinutes = 240,

    [string]$ReportPath,

    [switch]$AllowMissingLog,

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

$expectedExclusions = @(
    'powershell.exe',
    'pwsh.exe',
    'rclone.exe',
    'wsl.exe',
    'wslhost.exe',
    'vmmemWSL',
    'Docker Desktop.exe',
    'GoogleDriveFS.exe',
    'OneDrive*',
    'OneDrive.exe',
    'OneDrive.Sync.Service.exe',
    'FileSyncHelper.exe',
    'explorer.exe',
    'dwm.exe',
    'TextInputHost.exe',
    'TabTip.exe',
    'ProcessGovernor.exe',
    'ProcessLasso.exe',
    'ProcessLassoLauncher.exe',
    'ProcessLassoService.exe'
)

$expectedLoggingFlags = [ordered]@{
    IncludeCommandLines            = 'true'
    LogProcessExecutions          = 'true'
    LogProcessTerminations        = 'true'
    LogEfficiencyMode             = 'true'
    LogCPUSets                    = 'true'
    LogDefaultPriorityAdjustments = 'true'
    LogDefaultAffinityAdjustments = 'true'
    LogProBalanceBegin            = 'true'
    LogProBalanceEnd              = 'true'
    LogPowerProfileChanges        = 'true'
    LogSmartTrim                  = 'true'
    LogCPULimiter                 = 'true'
}

function Read-ProcessLassoIni {
    param([Parameter(Mandatory)] [string]$Path)

    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding Unicode -ErrorAction Stop
    } catch {
        return Get-Content -LiteralPath $Path -Raw -ErrorAction Stop
    }
}

function ConvertFrom-IniText {
    param([Parameter(Mandatory)] [string]$Text)

    $sections = @{}
    $currentSection = 'global'
    $sections[$currentSection] = @{}

    foreach ($rawLine in ($Text -split "`r?`n")) {
        $line = $rawLine.Trim().Trim([char]0xFEFF)
        if (-not $line -or $line.StartsWith(';') -or $line.StartsWith('#')) {
            continue
        }

        if ($line.StartsWith('[') -and $line.EndsWith(']')) {
            $currentSection = $line.Trim('[', ']')
            if (-not $sections.ContainsKey($currentSection)) {
                $sections[$currentSection] = @{}
            }
            continue
        }

        $parts = $line -split '=', 2
        if ($parts.Count -eq 2) {
            $sections[$currentSection][$parts[0].Trim()] = $parts[1].Trim()
        }
    }

    return $sections
}

function Get-IniValue {
    param(
        [Parameter(Mandatory)] [hashtable]$Sections,
        [Parameter(Mandatory)] [string]$Section,
        [Parameter(Mandatory)] [string]$Key
    )

    if ($Sections.ContainsKey($Section) -and $Sections[$Section].ContainsKey($Key)) {
        return $Sections[$Section][$Key]
    }

    return ''
}

function Get-CommaList {
    param([AllowNull()] [string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return @()
    }

    return @($Value -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Test-ListContains {
    param(
        [AllowNull()] [object[]]$List,
        [Parameter(Mandatory)] [string]$Expected
    )

    if (-not $List) {
        return $false
    }

    foreach ($item in $List) {
        if ($item -ieq $Expected) {
            return $true
        }
    }

    return $false
}

function Add-Failure {
    param(
        [System.Collections.IList]$Failures,
        [Parameter(Mandatory)] [string]$Message
    )

    [void]$Failures.Add($Message)
    Write-Error -Message $Message -ErrorAction Continue
}

$failures = [System.Collections.Generic.List[string]]::new()

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Add-Failure -Failures $failures -Message "Process Lasso config not found: $ConfigPath"
    $sections = @{}
} else {
    $sections = ConvertFrom-IniText -Text (Read-ProcessLassoIni -Path $ConfigPath)
}

$governor = Get-Process -Name 'ProcessGovernor' -ErrorAction SilentlyContinue | Select-Object -First 1
$governorStatus = if ($governor) {
    [pscustomobject]@{
        Running    = $true
        Id         = $governor.Id
        Responding = $governor.Responding
        StartTime  = $(try { $governor.StartTime } catch { $null })
        Path       = $(try { $governor.Path } catch { $null })
    }
} else {
    [pscustomobject]@{
        Running    = $false
        Id         = $null
        Responding = $false
        StartTime  = $null
        Path       = $null
    }
}

if (-not $governorStatus.Running) {
    Add-Failure -Failures $failures -Message 'ProcessGovernor.exe is not running.'
} elseif ($governorStatus.Responding -eq $false) {
    Add-Failure -Failures $failures -Message "ProcessGovernor.exe is running but not responding (PID $($governorStatus.Id))."
}

$oocExclusions = @(Get-CommaList (Get-IniValue -Sections $sections -Section 'OutOfControlProcessRestraint' -Key 'OocExclusions'))
$smartTrimExclusions = @(Get-CommaList (Get-IniValue -Sections $sections -Section 'MemoryManagement' -Key 'SmartTrimExclusions'))

$missingOoc = @($expectedExclusions | Where-Object { -not (Test-ListContains -List $oocExclusions -Expected $_) })
$missingSmartTrim = @($expectedExclusions | Where-Object { -not (Test-ListContains -List $smartTrimExclusions -Expected $_) })
if ($missingOoc.Count -gt 0) {
    Add-Failure -Failures $failures -Message "Missing ProBalance exclusions: $($missingOoc -join ', ')"
}
if ($missingSmartTrim.Count -gt 0) {
    Add-Failure -Failures $failures -Message "Missing SmartTrim exclusions: $($missingSmartTrim -join ', ')"
}

$loggingResults = @(foreach ($flag in $expectedLoggingFlags.GetEnumerator()) {
    $actual = Get-IniValue -Sections $sections -Section 'Logging' -Key $flag.Key
    $ok = $actual -ieq $flag.Value
    if (-not $ok) {
        Add-Failure -Failures $failures -Message "Logging flag $($flag.Key) expected '$($flag.Value)' but found '$actual'."
    }
    [pscustomobject]@{
        Key      = $flag.Key
        Expected = $flag.Value
        Actual   = $actual
        Ok       = $ok
    }
})

$logLines = @()
if (Test-Path -LiteralPath $LogPath) {
    $cutoff = (Get-Date).AddMinutes(-1 * [Math]::Max(1, $LookbackMinutes))
    $logLines = @(
        Get-Content -LiteralPath $LogPath -Tail 1000 -ErrorAction SilentlyContinue |
            Where-Object {
                $line = $_.Trim('"')
                $fields = $line -split '","'
                if ($fields.Count -lt 2) {
                    $true
                    return
                }
                [datetime]$timestamp = [datetime]::MinValue
                if ([datetime]::TryParseExact($fields[1], 'yyyy-MM-dd HH:mm:ss', [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::None, [ref]$timestamp)) {
                    $timestamp -ge $cutoff
                    return
                }
                $true
            } |
            Select-Object -Last 80
    )
} elseif (-not $AllowMissingLog) {
    Add-Failure -Failures $failures -Message "Process Lasso log not found: $LogPath"
}

$result = [pscustomobject]@{
    Ok                    = ($failures.Count -eq 0)
    DryRun                = [bool]$DryRun
    GeneratedAt           = (Get-Date).ToString('o')
    ConfigPath            = $ConfigPath
    LogPath               = $LogPath
    LookbackMinutes       = $LookbackMinutes
    Governor              = $governorStatus
    MissingOocExclusions  = $missingOoc
    MissingSmartTrimItems = $missingSmartTrim
    Logging               = @($loggingResults)
    RecentLogLines        = $logLines
    Failures              = @($failures)
}

if ($ReportPath -and -not $DryRun) {
    $reportDirectory = Split-Path -Parent $ReportPath
    if ($reportDirectory) {
        New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
    }
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ReportPath -Encoding UTF8
}

$failedLoggingResults = @($loggingResults | Where-Object { -not $_.Ok })
$recentLogLines = @($logLines)
$summary = @(
    [pscustomobject]@{ Check = 'GovernorRunning'; Ok = $governorStatus.Running; Detail = $governorStatus.Id }
    [pscustomobject]@{ Check = 'GovernorResponding'; Ok = [bool]$governorStatus.Responding; Detail = $governorStatus.Responding }
    [pscustomobject]@{ Check = 'ProBalanceExclusions'; Ok = ($missingOoc.Count -eq 0); Detail = $(if ($missingOoc.Count) { $missingOoc -join ', ' } else { 'ok' }) }
    [pscustomobject]@{ Check = 'SmartTrimExclusions'; Ok = ($missingSmartTrim.Count -eq 0); Detail = $(if ($missingSmartTrim.Count) { $missingSmartTrim -join ', ' } else { 'ok' }) }
    [pscustomobject]@{ Check = 'LoggingFlags'; Ok = ($failedLoggingResults.Count -eq 0); Detail = (($failedLoggingResults | Select-Object -ExpandProperty Key) -join ', ') }
    [pscustomobject]@{ Check = 'RecentLogLines'; Ok = ($recentLogLines.Count -gt 0 -or $AllowMissingLog); Detail = $recentLogLines.Count }
)

$summary | Format-Table -AutoSize
$result

if ($failures.Count -gt 0) {
    exit 1
}
