#Requires -Version 7.2

<#
.SYNOPSIS
Deploys the public PowerShell profile shim to this user's profile roots, with a backup of every file it replaces.
.DESCRIPTION
`Config/PowerShell/Profile/Microsoft.PowerShell_profile.ps1` is the Git-backed shim. PowerShell
loads it from `$PROFILE.CurrentUserCurrentHost`, which is the OneDrive Documents folder when Known
Folder Move is on and `~\Documents\PowerShell` otherwise, and a different tool or token can resolve
the other one. `Invoke-PcaiToolingHealth.ps1` reports any root whose copy differs from the source
(`shim=DRIFT`). This script is the repair: it writes the source bytes to each root that exists.

For each target:
- identical to the source (line endings ignored): left alone (`Unchanged`);
- missing, and its folder is missing too (for example no OneDrive): left alone (`Skipped`);
- otherwise replaced (`Replaced`, or `Created` when only the file was missing). A replaced file is
  first copied to the backup directory as `<host>.<folder>.<file>.bak-<sha256 prefix>`, and one
  JSON line in `receipts.jsonl` there records the UTC time, the target, both hashes and the
  backup path. File names carry a hash rather than a date; the time is in the receipt.

The replacement is staged next to the target, parsed, then moved into place, so a reader never sees
a partial file, and a source that does not parse is refused before anything changes. The private
profile (`~/.config/powershell`) is machine-local by design and is never touched here.
.PARAMETER SourcePath
The shim to deploy. Defaults to the copy in the checkout this script runs from.
.PARAMETER Target
Files to deploy to. Defaults to the Documents and OneDrive Documents PowerShell folders.
.PARAMETER BackupRoot
Where displaced files and the receipt go. Defaults to `~/.machine/backups/profiles`.
.PARAMETER PassThru
Emit one result object per target.
.EXAMPLE
pwsh -NoProfile -File Tools/SystemScripts/Machine/Install-PcaiProfileShim.ps1 -WhatIf
.EXAMPLE
pwsh -NoProfile -File Tools/SystemScripts/Machine/Install-PcaiProfileShim.ps1
#>
[CmdletBinding(SupportsShouldProcess = $true, PositionalBinding = $false)]
param(
    [string]$SourcePath,
    [string[]]$Target,
    [string]$BackupRoot,
    [switch]$PassThru,
    [Alias('h')][switch]$Help,
    # PowerShell cannot alias `--help`; it arrives here and is honored below (AGENTS.md: -h, --help).
    [Parameter(ValueFromRemainingArguments, DontShow)][string[]]$RemainingArguments = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-PcaiShimDefaultTarget {
    <# The shim locations pwsh can load for this user. Pure apart from the Documents lookup. #>
    [CmdletBinding()]
    param()
    $leaf = 'PowerShell\Microsoft.PowerShell_profile.ps1'
    $roots = [Collections.Generic.List[string]]::new()
    foreach ($r in @([Environment]::GetFolderPath('MyDocuments'), (Join-Path $HOME 'Documents'), (Join-Path $HOME 'OneDrive\Documents'))) {
        if ($r -and $r -notin $roots) { $roots.Add($r) }
    }
    return , @($roots | ForEach-Object { Join-Path $_ $leaf })
}

function Get-PcaiShimNormalizedText {
    <# File text with line endings normalized, for comparing a CRLF checkout with an LF copy. #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    return ([IO.File]::ReadAllText($Path) -replace "`r`n", "`n").TrimEnd()
}

function Test-PcaiShimParses {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    $tokens = $null; $errors = $null
    [void][Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    return (@($errors).Count -eq 0)
}

function Install-PcaiProfileShim {
    [CmdletBinding(SupportsShouldProcess = $true)]
    param([Parameter(Mandatory)][string]$Source, [Parameter(Mandatory)][string[]]$Path, [Parameter(Mandatory)][string]$Backup)
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) { throw "Shim source not found: $Source" }
    if (-not (Test-PcaiShimParses -Path $Source)) { throw "Shim source does not parse; nothing was changed: $Source" }
    $sourceBytes = [IO.File]::ReadAllBytes($Source)
    $sourceHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($sourceBytes)).ToLowerInvariant()
    $sourceText = Get-PcaiShimNormalizedText -Path $Source
    $hostName = ([Environment]::MachineName).ToLowerInvariant()
    foreach ($target in ($Path | Select-Object -Unique)) {
        $result = [ordered]@{ Target = $target; Action = $null; Backup = $null; Note = $null }
        $exists = Test-Path -LiteralPath $target -PathType Leaf
        $folder = Split-Path -Parent $target
        if ($exists -and (Get-PcaiShimNormalizedText -Path $target) -ceq $sourceText) {
            $result.Action = 'Unchanged'
        } elseif (-not $exists -and -not (Test-Path -LiteralPath $folder -PathType Container) -and -not (Test-Path -LiteralPath (Split-Path -Parent $folder) -PathType Container)) {
            $result.Action = 'Skipped'; $result.Note = 'its Documents root does not exist on this host'
        } elseif (-not $PSCmdlet.ShouldProcess($target, $(if ($exists) { 'replace the profile shim with a backup' } else { 'create the profile shim' }))) {
            $result.Action = 'WouldChange'
        } else {
            $null = New-Item -ItemType Directory -Force -Path $folder
            $oldHash = $null
            if ($exists) {
                $oldHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
                $null = New-Item -ItemType Directory -Force -Path $Backup
                $label = ((@($folder -split '[\\/]+' | Where-Object { $_ }) | Select-Object -Last 3) -join ".") -replace "[^A-Za-z0-9._-]", "_"
                $backupPath = Join-Path $Backup "$hostName.$label.$(Split-Path -Leaf $target).bak-$($oldHash.Substring(0, 12))"
                if (-not (Test-Path -LiteralPath $backupPath)) { Copy-Item -LiteralPath $target -Destination $backupPath }
                $result.Backup = $backupPath
            }
            $stage = Join-Path $folder ".pcai-shim-$PID.tmp"
            try {
                [IO.File]::WriteAllBytes($stage, $sourceBytes)
                if (-not (Test-PcaiShimParses -Path $stage)) { throw "Staged shim does not parse: $stage" }
                Move-Item -LiteralPath $stage -Destination $target -Force
            } finally { if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Force } }
            $newHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($newHash -ne $sourceHash) { throw "Deployed shim hash $newHash differs from the source $sourceHash at $target" }
            $result.Action = if ($exists) { 'Replaced' } else { 'Created' }
            $receipt = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); host = $hostName; target = $target; action = $result.Action; oldSha256 = $oldHash; newSha256 = $newHash; backup = $result.Backup; source = $Source }
            $null = New-Item -ItemType Directory -Force -Path $Backup
            Add-Content -LiteralPath (Join-Path $Backup 'receipts.jsonl') -Value ($receipt | ConvertTo-Json -Compress) -Encoding utf8NoBOM
        }
        [pscustomobject]$result
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    $unknown = @($RemainingArguments | Where-Object { $_ -ne '--help' })
    if ($unknown) { throw "Unknown arguments: $($unknown -join ' '). Use -h or --help." }
    if ($Help -or $RemainingArguments -contains '--help') { Get-Help $PSCommandPath -Detailed; return }
    if (-not $SourcePath) { $SourcePath = Join-Path $PSScriptRoot '../../../Config/PowerShell/Profile/Microsoft.PowerShell_profile.ps1' }
    if (-not $Target) { $Target = Get-PcaiShimDefaultTarget }
    if (-not $BackupRoot) { $BackupRoot = Join-Path $HOME '.machine/backups/profiles' }
    $results = @(Install-PcaiProfileShim -Source $SourcePath -Path $Target -Backup $BackupRoot -WhatIf:$WhatIfPreference)
    foreach ($r in $results) { Write-Host ("{0,-10} {1}{2}{3}" -f $r.Action, $r.Target, $(if ($r.Backup) { "  (backup $($r.Backup))" } else { '' }), $(if ($r.Note) { "  ($($r.Note))" } else { '' })) }
    if ($PassThru) { $results }
}
