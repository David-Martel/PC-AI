#Requires -Version 5.1
<#
.SYNOPSIS
Archives orphaned cloud-sync folders on the F: cloud-cache-disk.

.DESCRIPTION
Moves stale/orphaned cloud client folders on F: into F:\_archive using an
in-volume NTFS rename (metadata-only: instant, no data copied, no extra space).

Nothing is deleted. Every move is reversible by renaming back. The script
refuses to run if F: is not the expected cloud-cache-disk volume, and skips any
folder whose sync client is still actively using it.

.PARAMETER WhatIf
Show what would move without moving it.

.PARAMETER DryRun
Non-mutating preview: report every planned move without performing it. The long
CLI form `--DryRun` is also accepted. Equivalent to -WhatIf, exposed under the
name the repo's session-script contract requires.

.PARAMETER Help
Print script help and exit. The aliases `-h` and `--help` are also accepted.

.EXAMPLE
pwsh -File .\Tools\Restore-CloudCacheLayout.ps1 -WhatIf
.EXAMPLE
pwsh -File .\Tools\Restore-CloudCacheLayout.ps1 -DryRun
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ArchiveRoot = 'F:\_archive',
    [string]$Stamp = (Get-Date -Format 'yyyyMMdd'),
    [switch]$DryRun,
    [Alias('h', '?')]
    [switch]$Help,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# CLI contract (AGENTS.md): long forms first, and help BEFORE any environment
# guard -- otherwise `-h` on a machine without F: mounted would throw instead of
# printing help, which defeats the point of a help switch.
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
# Every mutation below is ShouldProcess-guarded, so forcing $WhatIfPreference is
# what makes -DryRun genuinely non-mutating rather than merely advisory.
if ($DryRun) { $WhatIfPreference = $true }

# --- Guard: F: must be the real cloud-cache-disk, not some other volume ---
$vol = Get-Volume -DriveLetter F -ErrorAction SilentlyContinue
if (-not $vol) { throw 'F: is not mounted - refusing to run.' }
if ($vol.FileSystemLabel -ne 'cloud-cache-disk') {
    throw "F: label is '$($vol.FileSystemLabel)', expected 'cloud-cache-disk' - refusing to run."
}
Write-Host "F: verified as cloud-cache-disk ($([math]::Round($vol.SizeRemaining/1GB)) GB free)" -ForegroundColor Green

# --- Active roots that must NEVER be archived ---
$activeRoots = @()
$dbxInfo = Join-Path $env:LOCALAPPDATA 'Dropbox\info.json'
if (Test-Path $dbxInfo) {
    $j = Get-Content $dbxInfo -Raw | ConvertFrom-Json
    foreach ($k in $j.PSObject.Properties.Name) {
        $acct = $j.$k
        foreach ($field in 'root_path', 'path') {
            $prop = $acct.PSObject.Properties[$field]
            if ($prop -and $prop.Value) { $activeRoots += [string]$prop.Value }
        }
    }
}
$od = Get-ItemProperty 'HKCU:\SOFTWARE\Microsoft\OneDrive\Accounts\*' -ErrorAction SilentlyContinue
foreach ($a in $od) {
    # not every OneDrive account key carries UserFolder (e.g. FileCoAuth)
    $uf = $a.PSObject.Properties['UserFolder']
    if ($uf -and $uf.Value) { $activeRoots += [string]$uf.Value }
}
Write-Host "Active sync roots detected (protected):" -ForegroundColor Cyan
$activeRoots | Sort-Object -Unique | ForEach-Object { Write-Host "    $_" }

# --- Candidates ---
$candidates = @(
    @{ Path = 'F:\University of Michigan Dropbox'; Name = "UMich-Dropbox-stale-$Stamp";
       Why  = 'orphaned Dropbox root - client now syncs to C:, placeholders unreadable' },
    @{ Path = 'F:\OneDrive - Auricle Inc'; Name = "OneDrive-AuricleInc-stale-$Stamp";
       Why  = 'OneDrive Business account no longer configured' },
    @{ Path = 'F:\Proton-Drive'; Name = "Proton-Drive-stale-$Stamp";
       Why  = 'Proton Drive has no configured sync root' }
)

if (-not (Test-Path $ArchiveRoot)) {
    if ($PSCmdlet.ShouldProcess($ArchiveRoot, 'Create archive root')) {
        New-Item -ItemType Directory -Path $ArchiveRoot -Force | Out-Null
    }
}

$results = @()
foreach ($c in $candidates) {
    $src = $c.Path
    if (-not (Test-Path $src)) {
        Write-Host "SKIP  $src (not present)" -ForegroundColor DarkGray
        $results += [pscustomobject]@{ Source = $src; Action = 'skip-missing'; Dest = '' }
        continue
    }

    # never touch a root a client is actively syncing
    $clash = $activeRoots | Where-Object {
        $_ -and ($src -eq $_.TrimEnd('\') -or $_.TrimEnd('\').StartsWith($src, 'OrdinalIgnoreCase'))
    }
    if ($clash) {
        Write-Warning "SKIP  $src - ACTIVE sync root ($clash)"
        $results += [pscustomobject]@{ Source = $src; Action = 'skip-active'; Dest = '' }
        continue
    }

    $dest = Join-Path $ArchiveRoot $c.Name
    if (Test-Path $dest) { throw "Destination already exists: $dest" }

    Write-Host "MOVE  $src" -ForegroundColor Yellow
    Write-Host "   -> $dest   ($($c.Why))"
    if ($PSCmdlet.ShouldProcess($src, "Rename to $dest")) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        Move-Item -LiteralPath $src -Destination $dest -Force
        $sw.Stop()
        Write-Host ("      done in {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
        $results += [pscustomobject]@{ Source = $src; Action = 'moved'; Dest = $dest }
    } else {
        $results += [pscustomobject]@{ Source = $src; Action = 'whatif'; Dest = $dest }
    }
}

Write-Host ""
Write-Host "=== RESULT ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
$volAfter = Get-Volume -DriveLetter F
Write-Host ("F: free now: {0:N0} GB (was {1:N0} GB)" -f ($volAfter.SizeRemaining / 1GB), ($vol.SizeRemaining / 1GB))
Write-Host "Reversal: Move-Item <dest> back to <source>. No data was copied or deleted."
