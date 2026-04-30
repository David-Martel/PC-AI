<#
.SYNOPSIS
Task Scheduler wrapper for the Gemini CLI update check.

.DESCRIPTION
The scheduled task previously pointed at
C:\Users\david\gemini-cli\update-scripts\check-releases.ps1, but that source
script was missing during the 2026-04-30 system-script migration. This wrapper
keeps the task repo-owned and fails loudly until the real Gemini updater is
restored or replaced.

.PARAMETER Channel
Gemini release channel name.

.PARAMETER DryRun
Report what would run without changing state. The long form --DryRun is also
accepted.

.PARAMETER Help
Print help and exit. The aliases -h and --help are also accepted.
#>
[CmdletBinding()]
param(
    [string]$Channel = 'stable',
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

$originalPath = 'C:\Users\david\gemini-cli\update-scripts\check-releases.ps1'
$eventSource = 'PC-AI-SystemScripts'

$result = [ordered]@{
    status        = 'missing_original_script'
    channel       = $Channel
    wrapper_path  = $PSCommandPath
    original_path = $originalPath
    dry_run       = [bool]$DryRun
    timestamp     = (Get-Date).ToString('o')
}

if ($DryRun) {
    $result | ConvertTo-Json -Depth 4
    return
}

try {
    if (-not [System.Diagnostics.EventLog]::SourceExists($eventSource)) {
        New-EventLog -LogName Application -Source $eventSource -ErrorAction SilentlyContinue
    }
    Write-EventLog -LogName Application -Source $eventSource -EventId 4710 -EntryType Warning -Message (
        "Gemini CLI updater task cannot run because the original script is missing: $originalPath"
    ) -ErrorAction SilentlyContinue
}
catch {
    Write-Warning "Unable to write Application event: $_"
}

$result | ConvertTo-Json -Depth 4
throw "Gemini CLI updater source script is missing: $originalPath"
